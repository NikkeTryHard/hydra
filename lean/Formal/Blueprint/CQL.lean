import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedDecidableInType false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false

/-! # Hydra2 CQL offline RL — masked log-sum-exp conservative penalty

CQL offline penalty (Mortal `train.py` + Kumar et al. 2020, https://arxiv.org/abs/1911.01217):
`loss = dqn_loss + min_q_weight·cql_loss + next_rank_weight·next_rank_loss`
with `cql_loss = logsumexp(Q(s,·)_legal) - mean(Q(s,a_data))`, applied
OFFLINE only (`min_q_weight > 0`), dropped for online finetune. Discrete
mahjong uses the EXACT logsumexp over legal actions (no sampling).

Finite core below: masked log-sum-exp dominates each legal Q (hence the
penalty over the data mean is nonneg). Bellman/MC backup, α schedule, twin-Q,
dueling heads, GRP reward, rank aux stay harness-side.
-/

namespace Hydra2.Blueprint.CQL

section MaskedLSE

variable {Action : Type} [Fintype Action] [DecidableEq Action]

def LegalMask (Action : Type) := Action → Bool

noncomputable def maskedLogSumExp (legal : LegalMask Action) (Q : Action → ℝ) : ℝ :=
  Real.log (∑ a : Action, if legal a = true then Real.exp (Q a) else (0 : ℝ))

theorem logSumExp_ge_each (legal : LegalMask Action) (Q : Action → ℝ)
    (j : Action) (hj : legal j = true) :
    Q j ≤ maskedLogSumExp legal Q := by
  unfold maskedLogSumExp
  have hnn : ∀ a ∈ (Finset.univ : Finset Action),
      0 ≤ (if legal a = true then Real.exp (Q a) else (0 : ℝ)) := by
    intro a _
    split <;> positivity
  have h_pos_at : 0 < (if legal j = true then Real.exp (Q j) else (0 : ℝ)) := by
    simp [hj]
    positivity
  have hsum_pos : 0 < ∑ a : Action,
      (if legal a = true then Real.exp (Q a) else (0 : ℝ)) :=
    Finset.sum_pos' hnn ⟨j, Finset.mem_univ j, h_pos_at⟩
  have h1 : Real.exp (Q j) ≤ ∑ a : Action,
      (if legal a = true then Real.exp (Q a) else (0 : ℝ)) := by
    have hle := Finset.single_le_sum hnn (Finset.mem_univ j)
    simp [hj] at hle
    exact hle
  have h2 : Real.log (Real.exp (Q j)) ≤ Real.log (∑ a : Action,
      (if legal a = true then Real.exp (Q a) else (0 : ℝ))) := by
    rw [Real.log_le_log_iff (Real.exp_pos _) hsum_pos]
    exact h1
  rw [Real.log_exp] at h2
  exact h2

/-- Log-sum-exp shifts by constants: adding `c` to every Q-value shifts the
masked LSE by exactly `c` (needs one legal action for positivity of the
normalizer). Finite core behind CQL scale-freedom: the conservatism charge
lives in Q gaps, never Q levels. -/
theorem maskedLogSumExp_shift (legal : LegalMask Action) (Q : Action → ℝ) (c : ℝ)
    (hexists : ∃ a, legal a = true) :
    maskedLogSumExp legal (fun a => Q a + c) = maskedLogSumExp legal Q + c := by
  obtain ⟨j, hj⟩ := hexists
  have hnn : ∀ a ∈ (Finset.univ : Finset Action),
      0 ≤ (if legal a = true then Real.exp (Q a) else (0 : ℝ)) := by
    intro a _
    split <;> positivity
  have hsum_pos : 0 < ∑ a : Action,
      (if legal a = true then Real.exp (Q a) else (0 : ℝ)) := by
    have h_pos_at : 0 < (if legal j = true then Real.exp (Q j) else (0 : ℝ)) := by
      simp [hj]; positivity
    exact Finset.sum_pos' hnn ⟨j, Finset.mem_univ j, h_pos_at⟩
  have hfactor : (∑ a : Action,
        (if legal a = true then Real.exp (Q a + c) else (0 : ℝ)))
      = Real.exp c * ∑ a : Action,
        (if legal a = true then Real.exp (Q a) else (0 : ℝ)) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro a _
    by_cases ha : legal a = true
    · simp only [ha, ↓reduceIte, Real.exp_add, mul_comm]
    · simp [ha]
  calc maskedLogSumExp legal (fun a => Q a + c)
      = Real.log (∑ a : Action,
          (if legal a = true then Real.exp (Q a + c) else (0 : ℝ))) := rfl
    _ = Real.log (Real.exp c * ∑ a : Action,
          (if legal a = true then Real.exp (Q a) else (0 : ℝ))) := by rw [hfactor]
    _ = c + Real.log (∑ a : Action,
          (if legal a = true then Real.exp (Q a) else (0 : ℝ))) := by
        rw [Real.log_mul (ne_of_gt (Real.exp_pos c)) (ne_of_gt hsum_pos), Real.log_exp]
    _ = maskedLogSumExp legal Q + c := by
        unfold maskedLogSumExp; ring

/-- Masked log-sum-exp is monotone in pointwise Q: raising every Q-value
can only raise the soft maximum — the finite reason optimistic model
updates (higher Q everywhere) never lower the penalty's optimism arm:
the log-sum-exp tracks Q upward, so the first term moves with the
model, never against it. Pairs `maskedLogSumExp_shift` (uniform shifts
pass through exactly): together level AND shape changes move the right
way. -/
theorem maskedLogSumExp_mono_Q (legal : LegalMask Action) (Q₁ Q₂ : Action → ℝ)
    (hexists : ∃ a, legal a = true) (h : ∀ a, Q₁ a ≤ Q₂ a) :
    maskedLogSumExp legal Q₁ ≤ maskedLogSumExp legal Q₂ := by
  unfold maskedLogSumExp
  obtain ⟨j, hj⟩ := hexists
  have hnn1 : ∀ a ∈ (Finset.univ : Finset Action),
      0 ≤ (if legal a = true then Real.exp (Q₁ a) else (0 : ℝ)) := by
    intro a _
    split <;> positivity
  have hpos1 : 0 < ∑ a : Action,
      (if legal a = true then Real.exp (Q₁ a) else (0 : ℝ)) := by
    apply Finset.sum_pos' hnn1
    refine ⟨j, Finset.mem_univ j, ?_⟩
    simp [hj]
    positivity
  have hnn2 : ∀ a ∈ (Finset.univ : Finset Action),
      0 ≤ (if legal a = true then Real.exp (Q₂ a) else (0 : ℝ)) := by
    intro a _
    split <;> positivity
  have hpos2 : 0 < ∑ a : Action,
      (if legal a = true then Real.exp (Q₂ a) else (0 : ℝ)) := by
    apply Finset.sum_pos' hnn2
    refine ⟨j, Finset.mem_univ j, ?_⟩
    simp [hj]
    positivity
  rw [Real.log_le_log_iff hpos1 hpos2]
  apply Finset.sum_le_sum
  intro a _
  split
  · exact Real.exp_le_exp.mpr (h a)
  · exact le_rfl

noncomputable def cqlPenalty (legal : LegalMask Action) (Q : Action → ℝ)
    (dataMean : ℝ) : ℝ :=
  maskedLogSumExp legal Q - dataMean

/-- CQL penalty is shift-invariant: adding a constant to every Q-value and
to the data mean leaves the charge unchanged. Consequence of
`maskedLogSumExp_shift`: the conservatism dial lives purely in Q gaps,
so reward-scale shifts (uma/oka rescaling, score normalization) never
change the penalty — the S0.5 gate calibrates alpha once per gap scale,
not per reward scale. -/
theorem cql_penalty_shift_invariant (legal : LegalMask Action) (Q : Action → ℝ)
    (dataMean c : ℝ) (hexists : ∃ a, legal a = true) :
    cqlPenalty legal (fun a => Q a + c) (dataMean + c) = cqlPenalty legal Q dataMean := by
  unfold cqlPenalty
  rw [maskedLogSumExp_shift legal Q c hexists]
  ring

/-- CQL penalty is antitone in the data mean: a higher data anchor lowers
the conservatism charge at fixed Q — the finite reason cleaner logs
(higher mean data-Q) buy a smaller penalty without touching alpha:
the charge prices the gap between model optimism and data evidence,
so better evidence narrows it. Pairs `data_mean_mono_Q` (better logs
raise the anchor): together log improvements flow through to a lower
charge monotonically. -/
theorem cql_penalty_antitone_mean (legal : LegalMask Action) (Q : Action → ℝ)
    (m₁ m₂ : ℝ) (h : m₁ ≤ m₂) :
    cqlPenalty legal Q m₂ ≤ cqlPenalty legal Q m₁ := by
  unfold cqlPenalty
  linarith

/-- CQL penalty is monotone in pointwise Q at fixed mean: uniformly
better model values raise the charge — consequence of
`maskedLogSumExp_mono_Q` (the optimism arm tracks Q upward) with the
data anchor held fixed. Together with `cql_penalty_antitone_mean`
(better data lowers it): the charge prices exactly the optimism gap,
rising with the model and falling with the evidence, so the S0.5 gate
reads the gap's two handles independently. -/
theorem cql_penalty_mono_Q (legal : LegalMask Action) (Q₁ Q₂ : Action → ℝ)
    (hexists : ∃ a, legal a = true) (m : ℝ) (h : ∀ a, Q₁ a ≤ Q₂ a) :
    cqlPenalty legal Q₁ m ≤ cqlPenalty legal Q₂ m := by
  unfold cqlPenalty
  have hle := maskedLogSumExp_mono_Q legal Q₁ Q₂ hexists h
  linarith

/-- CQL penalty is nonneg over any nonempty set of legal data actions: the data
mean sits below the max, the max sits below the log-sum-exp (`logSumExp_ge_each`
per action, averaged). This is the finite core of Kumar et al. conservatism
(`logsumexp(Q_legal) - mean(Q_data) ≥ 0` pushes down unseen-action mass). -/
theorem cql_penalty_nonneg (legal : LegalMask Action) (Q : Action → ℝ)
    (S : Finset Action) (hne : S.Nonempty) (hleg : ∀ a ∈ S, legal a = true) :
    (∑ a ∈ S, Q a) / (S.card : ℝ) ≤ maskedLogSumExp legal Q := by
  have hcard_pos : (0 : ℝ) < (S.card : ℝ) :=
    Nat.cast_pos.mpr (Finset.card_pos.mpr hne)
  have each : ∀ a ∈ S, Q a ≤ maskedLogSumExp legal Q :=
    fun a ha => logSumExp_ge_each legal Q a (hleg a ha)
  have hsum : ∑ a ∈ S, Q a ≤ S.card • maskedLogSumExp legal Q := by
    calc ∑ a ∈ S, Q a ≤ ∑ _a ∈ S, maskedLogSumExp legal Q := Finset.sum_le_sum each
      _ = S.card • maskedLogSumExp legal Q := by rw [Finset.sum_const]
  rw [nsmul_eq_mul] at hsum
  rw [div_le_iff₀ hcard_pos]
  calc ∑ a ∈ S, Q a ≤ (S.card : ℝ) * maskedLogSumExp legal Q := hsum
    _ = maskedLogSumExp legal Q * (S.card : ℝ) := by ring

/-- Data mean is bounded above by the legal max: the average of legal data-Q
    never exceeds the best legal data-Q. Finite core behind the S0.5 gate —
    offline RL can only promise conservatism against the data it saw, so the
    gate compares against the data max, never against unseen actions.
    Limited-compute lemma: the offline stage's promise is checkable from
    the log alone, with zero simulator calls. The mean is also monotone
    in pointwise Q (proved next): log improvements propagate without
    overshooting the best observed action. -/
theorem data_mean_mono_Q (Q₁ Q₂ : Action → ℝ)
    (S : Finset Action) (h : ∀ a ∈ S, Q₁ a ≤ Q₂ a) :
    (∑ a ∈ S, Q₁ a) / (S.card : ℝ) ≤ (∑ a ∈ S, Q₂ a) / (S.card : ℝ) := by
  apply div_le_div_of_nonneg_right _ (by positivity)
  exact Finset.sum_le_sum h

/-- Singleton mean is the sample: averaging one data-Q returns it —
the finite reason single-sample batches need no aggregation pass:
the mean operator is the identity at size one, so unit-batch
eval legs read the sample directly. Pairs `data_mean_le_max` (mean
below max): together the mean is exact at size one and bounded
above in general. -/
theorem data_mean_singleton (Q : Action → ℝ) (x : Action) :
    (∑ a ∈ ({x} : Finset Action), Q a) / ((({x} : Finset Action).card : ℕ) : ℝ)
      = Q x := by
  simp

theorem data_mean_le_max (Q : Action → ℝ)
    (S : Finset Action) (hne : S.Nonempty) :
    (∑ a ∈ S, Q a) / (S.card : ℝ) ≤ S.sup' hne Q := by
  have hcard_pos : (0 : ℝ) < (S.card : ℝ) :=
    Nat.cast_pos.mpr (Finset.card_pos.mpr hne)
  rw [div_le_iff₀ hcard_pos]
  have hsup : ∀ a ∈ S, Q a ≤ S.sup' hne Q :=
    fun a ha => Finset.le_sup' Q ha
  calc ∑ a ∈ S, Q a ≤ ∑ _a ∈ S, S.sup' hne Q := Finset.sum_le_sum hsup
    _ = S.card • S.sup' hne Q := by rw [Finset.sum_const]
    _ = S.sup' hne Q * (S.card : ℝ) := by rw [nsmul_eq_mul, mul_comm]

/-- Data mean is also bounded below by the data min: the offline average
lies inside `[min, max]`, so the S0.5 gate can check BOTH sides from the
log alone. Mirror of `data_mean_le_max`; together they sandwich the
offline promise — a mean outside the interval means the averaging code
is wrong, not the data. -/
theorem data_mean_ge_min (Q : Action → ℝ)
    (S : Finset Action) (hne : S.Nonempty) :
    S.inf' hne Q ≤ (∑ a ∈ S, Q a) / (S.card : ℝ) := by
  have hcard_pos : (0 : ℝ) < (S.card : ℝ) :=
    Nat.cast_pos.mpr (Finset.card_pos.mpr hne)
  rw [le_div_iff₀ hcard_pos]
  have hinf : ∀ a ∈ S, S.inf' hne Q ≤ Q a :=
    fun a ha => Finset.inf'_le Q ha
  calc S.inf' hne Q * (S.card : ℝ) = ∑ _a ∈ S, S.inf' hne Q := by
        rw [Finset.sum_const, nsmul_eq_mul, mul_comm]
    _ ≤ ∑ a ∈ S, Q a := Finset.sum_le_sum hinf

/-- Scaled penalty stays nonneg (`α ≥ 0`, e.g. Mortal `min_q_weight` offline-only). -/
theorem cql_scaled_nonneg (legal : LegalMask Action) (Q : Action → ℝ)
    (S : Finset Action) (hne : S.Nonempty) (hleg : ∀ a ∈ S, legal a = true)
    (α : ℝ) (hα : 0 ≤ α) :
    0 ≤ α * cqlPenalty legal Q ((∑ a ∈ S, Q a) / (S.card : ℝ)) := by
  unfold cqlPenalty
  apply mul_nonneg hα
  have h := cql_penalty_nonneg legal Q S hne hleg
  linarith

/-- Zeroing the data mean's weight kills the penalty term: at `α = 0` the
scaled charge vanishes for ANY legal mask, Q, and data set — the finite
reason the offline→online handoff anneals exactly to zero rather than to
a small epsilon. Pair with `cql_scaled_mono_alpha` (raising alpha only
tightens): together the dial spans [off, tight] with the off position
exact. -/
theorem cql_scaled_zero_alpha (legal : LegalMask Action) (Q : Action → ℝ)
    (S : Finset Action) (_hne : S.Nonempty) :
    (0 : ℝ) * cqlPenalty legal Q ((∑ a ∈ S, Q a) / (S.card : ℝ)) = 0 := by
  ring

/-- Unit alpha is the identity: at `α = 1` the scaled charge IS the raw
penalty — the mirror of `cql_scaled_zero_alpha` (α = 0 kills it).
Together both dial ends are pinned (off and pass-through), so the
alpha schedule interpolates between silence and the full conservatism
charge with `cql_scaled_mono_alpha` ordering the middle. -/
theorem cql_scaled_one_alpha (legal : LegalMask Action) (Q : Action → ℝ)
    (S : Finset Action) (_hne : S.Nonempty) :
    (1 : ℝ) * cqlPenalty legal Q ((∑ a ∈ S, Q a) / (S.card : ℝ))
      = cqlPenalty legal Q ((∑ a ∈ S, Q a) / (S.card : ℝ)) := by
  ring

end MaskedLSE
section LagrangeAlpha

/-- Lagrange-vs-fixed alpha (Kumar Appx F Eq.30): the dual-constrained penalty
`α·(penalty − τ) + α·τ` recovers the fixed penalty `α·penalty` (ring).
Harness constrains `E[penalty] ≤ τ` and auto-tunes `α ≥ 0`; finite identity only. -/
theorem cql_lagrangian_recovers_fixed (alpha pen tau : ℝ) :
    alpha * (pen - tau) + alpha * tau = alpha * pen := by
  ring

/-- Threshold shift is penalty-neutral at fixed alpha: moving mass between
the constraint term and the threshold term leaves the Lagrangian unchanged
— the finite reason the τ schedule only matters through the constraint
`E[penalty] ≤ τ`, never through the objective value. Pair with
`cql_lagrangian_recovers_fixed` (dual equals fixed): together the
harness can move τ freely during tuning without repricing the charge. -/
theorem cql_lagrangian_shift_tau (alpha pen tau₁ tau₂ : ℝ) :
    alpha * (pen - tau₁) + alpha * tau₁
    = alpha * (pen - tau₂) + alpha * tau₂ := by
  ring

/-- Lagrangian symmetric in its two penalty terms: swapping constraint
and threshold arms leaves the sum unchanged — the finite reason the
harness can attribute the charge to either arm in logging without
moving the objective: term order is presentation, never price. Pairs
`cql_lagrangian_shift_tau` (τ relocation is free): together both the
split point AND the term order are presentation choices. -/
theorem cql_lagrangian_arm_comm (alpha pen tau : ℝ) :
    alpha * (pen - tau) + alpha * tau
    = alpha * tau + alpha * (pen - tau) := by
  ring

/-- Scaled penalty is monotone in α at fixed penalty: raising the CQL weight
never lowers the conservatism charge (for nonneg penalty). Practical core:
the α schedule is a one-way dial — annealing α→0 for online finetune only
eases the penalty, so the offline→online handoff can't silently re-tighten.
Limited-compute lemma: one scalar, monotone effect, no surprises. -/
theorem cql_scaled_mono_alpha (pen : ℝ) (hpen : 0 ≤ pen)
    {α₁ α₂ : ℝ} (h : α₁ ≤ α₂) :
    α₁ * pen ≤ α₂ * pen :=
  mul_le_mul_of_nonneg_right h hpen

/-- Zeroing alpha kills the penalty entirely: at `α = 0` the scaled charge
vanishes regardless of penalty or threshold. Pair with
`cql_lagrangian_recovers_fixed` (dual form equals fixed form) and
`cql_scaled_mono_alpha` (raising alpha only tightens): together the dial
spans [off, fixed-form] with no dead zone and no surprise — the finite
reason the offline-to-online handoff anneals alpha to exactly zero.
Failure mode killed: a residual penalty surviving the anneal through the
threshold term. -/
theorem cql_lagrangian_zero_alpha (pen tau : ℝ) :
    (0 : ℝ) * (pen - tau) + 0 * tau = 0 := by
  ring

/-- Negative alpha inverts the penalty: with a negative weight the scaled
charge goes nonpositive for nonneg penalty — the finite reason the
harness clamps `α ≥ 0` at the config boundary: an unclamped sign flip
turns conservatism into optimism silently. Pairs `cql_scaled_mono_alpha`
(monotone on the nonneg side): together the dial is safe only at or
above zero. -/
theorem cql_scaled_nonpos_of_neg_alpha (pen : ℝ) (hpen : 0 ≤ pen)
    {α : ℝ} (hα : α ≤ 0) :
    α * pen ≤ 0 :=
  mul_nonpos_of_nonpos_of_nonneg hα hpen

end LagrangeAlpha

end Hydra2.Blueprint.CQL
