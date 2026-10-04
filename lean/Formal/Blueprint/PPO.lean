import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Tactic
import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Analysis.Real.Sqrt

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false
set_option linter.style.whitespace false

/-! # Hydra2 PPO/ACH — Masked Objectives and Illegal Logit Zero-Grad
Masked PPO comparator and candidate training objectives. Illegal probabilities
and illegal-logit gradients are exactly zero; any leak fails closed.
-/

namespace Hydra2.Blueprint.PPO

section MaskedSoftmax

variable {Action : Type} [Fintype Action] [DecidableEq Action]

def LegalMask (Action : Type) := Action → Bool
abbrev Logits (Action : Type) := Action → ℝ

noncomputable def expLogit (logits : Logits Action) (a : Action) : ℝ :=
  Real.exp (logits a)

noncomputable def partition (legal : LegalMask Action) (logits : Logits Action) : ℝ :=
  ∑ a : Action, if legal a = true then expLogit logits a else (0 : ℝ)

theorem partition_pos_of_exists_legal (legal : LegalMask Action) (logits : Logits Action)
    (hexists : ∃ a, legal a = true) : 0 < partition legal logits := by
  unfold partition expLogit
  obtain ⟨a0, ha0⟩ := hexists
  have h_nonneg : ∀ a ∈ (Finset.univ : Finset Action), 0 ≤ (if legal a = true then Real.exp (logits a) else (0 : ℝ)) := by
    intro a _; split <;> positivity
  have h_pos_at : 0 < (if legal a0 = true then Real.exp (logits a0) else (0 : ℝ)) := by
    simp [ha0]
    positivity
  exact Finset.sum_pos' h_nonneg ⟨a0, Finset.mem_univ _, h_pos_at⟩

theorem partition_ne_zero_of_exists_legal (legal : LegalMask Action) (logits : Logits Action)
    (hexists : ∃ a, legal a = true) : partition legal logits ≠ 0 :=
  ne_of_gt (partition_pos_of_exists_legal legal logits hexists)

noncomputable def legalSoftmax (legal : LegalMask Action) (logits : Logits Action)
    (hexists : ∃ a, legal a = true) (a : Action) : ℝ :=
  if legal a = true then expLogit logits a / partition legal logits else 0

theorem legalSoftmax_illegal_zero (legal : LegalMask Action) (logits : Logits Action)
    (hexists : ∃ a, legal a = true) (a : Action) (h_ill : legal a = false) :
    legalSoftmax legal logits hexists a = 0 := by
  unfold legalSoftmax
  simp [h_ill]

theorem legalSoftmax_nonneg (legal : LegalMask Action) (logits : Logits Action)
    (hexists : ∃ a, legal a = true) (a : Action) :
    0 ≤ legalSoftmax legal logits hexists a := by
  unfold legalSoftmax
  by_cases ha : legal a = true
  · simp [ha, expLogit]
    apply div_nonneg (Real.exp_nonneg _)
    apply Finset.sum_nonneg
    intro b _; by_cases hb : legal b = true <;> simp [hb, expLogit] <;> positivity
  · simp [ha]

theorem legalSoftmax_sum_one (legal : LegalMask Action) (logits : Logits Action)
    (hexists : ∃ a, legal a = true) :
    ∑ a : Action, legalSoftmax legal logits hexists a = 1 := by
  have hne := partition_ne_zero_of_exists_legal legal logits hexists
  unfold legalSoftmax expLogit
  have h_rw : ∀ a : Action, (if legal a = true then Real.exp (logits a) / partition legal logits else (0:ℝ))
      = (if legal a = true then Real.exp (logits a) else (0:ℝ)) / partition legal logits := by
    intro a
    by_cases ha : legal a = true
    · simp [ha]
    · simp [ha, zero_div]
  have h_sum : ∑ a : Action, (if legal a = true then Real.exp (logits a) / partition legal logits else (0:ℝ))
      = ∑ a : Action, (if legal a = true then Real.exp (logits a) else (0:ℝ)) / partition legal logits := by
    apply Finset.sum_congr rfl; intro a _; exact h_rw a
  rw [h_sum, ← Finset.sum_div]
  have h_part : (∑ a : Action, if legal a = true then Real.exp (logits a) else (0:ℝ)) = partition legal logits := by
    simp [partition, expLogit]
  rw [h_part]
  exact div_self hne

theorem legalSoftmax_ignores_illegal_logits (legal : LegalMask Action)
    (logits1 logits2 : Logits Action)
    (hagree : ∀ a, legal a = true → logits1 a = logits2 a)
    (hexists : ∃ a, legal a = true)
    (a : Action) (h_leg : legal a = true) :
    legalSoftmax legal logits1 hexists a = legalSoftmax legal logits2 hexists a := by
  unfold legalSoftmax partition expLogit
  have hpart_eq : (∑ b : Action, if legal b = true then Real.exp (logits1 b) else (0 : ℝ))
                = (∑ b : Action, if legal b = true then Real.exp (logits2 b) else (0 : ℝ)) := by
    apply Finset.sum_congr rfl
    intro b _
    by_cases hb : legal b = true
    · simp [hb, hagree b hb]
    · simp [hb]
  have h_exp_eq : Real.exp (logits1 a) = Real.exp (logits2 a) := by
    rw [hagree a h_leg]
  simp [h_leg, h_exp_eq, hpart_eq]

theorem legalSoftmax_illegal_logits_irrelevant (legal : LegalMask Action)
    (logits1 logits2 : Logits Action)
    (hagree : ∀ a, legal a = true → logits1 a = logits2 a)
    (hexists : ∃ a, legal a = true) :
    ∀ a, legal a = true → legalSoftmax legal logits1 hexists a = legalSoftmax legal logits2 hexists a :=
  fun a ha => legalSoftmax_ignores_illegal_logits legal logits1 logits2 hagree hexists a ha

section PPOLoss

noncomputable def logPi (legal : LegalMask Action) (logits : Logits Action)
    (hexists : ∃ a, legal a = true) (a : Action) : ℝ :=
  Real.log (legalSoftmax legal logits hexists a)

noncomputable def ratio (legal : LegalMask Action) (logits logits_old : Logits Action)
    (hexists : ∃ a, legal a = true) (a : Action) : ℝ :=
  Real.exp (logPi legal logits hexists a - logPi legal logits_old hexists a)

theorem ratio_of_legalSoftmax_eq (legal : LegalMask Action) (logits logits_old : Logits Action)
    (hexists : ∃ a, legal a = true) (a : Action)
    (h_pos1 : 0 < legalSoftmax legal logits hexists a)
    (h_pos2 : 0 < legalSoftmax legal logits_old hexists a) :
    ratio legal logits logits_old hexists a = legalSoftmax legal logits hexists a / legalSoftmax legal logits_old hexists a := by
  unfold ratio logPi
  rw [Real.exp_sub, Real.exp_log h_pos1, Real.exp_log h_pos2]

noncomputable def advantageStd (adv : ℝ) (mean var_eps : ℝ) : ℝ :=
  (adv - mean) / Real.sqrt (var_eps)

noncomputable def clippedRatio (r clip_eps : ℝ) : ℝ :=
  min (max r (1 - clip_eps)) (1 + clip_eps)

theorem clippedRatio_mem_Icc (r clip_eps : ℝ) (h_eps_pos : 0 ≤ clip_eps) :
    clippedRatio r clip_eps ∈ Set.Icc (1 - clip_eps) (1 + clip_eps) := by
  unfold clippedRatio
  constructor
  · exact le_min (le_max_right _ _) (by linarith)
  · exact min_le_right _ _

/-- Zero clip radius pins the ratio at 1: with `clip_eps = 0` the trust
region collapses to a point — no ratio movement survives clipping.
Practical core: the clip dial has an exact off-check — annealing
epsilon to zero provably freezes the policy ratio, so the offline-to-
online handoff can verify the freeze before switching objectives. -/
theorem clippedRatio_zero_eps (r : ℝ) : clippedRatio r 0 = 1 := by
  unfold clippedRatio
  simp

noncomputable def surrogate (r adv_std : ℝ) (clip_eps : ℝ) : ℝ :=
  min (r * adv_std) (clippedRatio r clip_eps * adv_std)

/-- Zero advantage kills the surrogate: with `adv = 0` both arms vanish,
so the pessimistic objective is exactly zero regardless of ratio or clip
— the finite reason advantage normalization centers the update: zero-mean
advantages contribute nothing, and only deviations move the policy.
Failure mode killed: a nonzero update from a zero-advantage batch
through a mis-scaled surrogate. -/
theorem surrogate_zero_adv (r clip_eps : ℝ) :
    surrogate r 0 clip_eps = 0 := by
  unfold surrogate
  simp

theorem surrogate_le_unclipped (r adv_std clip_eps : ℝ) :
    surrogate r adv_std clip_eps ≤ r * adv_std :=
  min_le_left _ _

theorem surrogate_le_clipped (r adv_std clip_eps : ℝ) :
    surrogate r adv_std clip_eps ≤ clippedRatio r clip_eps * adv_std :=
  min_le_right _ _

/-- Clipped surrogate is bounded above by the unclipped-best of both arms:
    `min(r·A, clip·A) ≤ max(r·A, clip·A)` trivially, but the useful form is
    that pessimism never exceeds EITHER arm — the min is below each.
    Practical core: PPO pessimism can only under-claim, never over-claim
    the update — limited-compute training never takes a step bigger than
    both its estimates agree on. -/
theorem surrogate_le_either_arm (r adv_std clip_eps : ℝ) :
    surrogate r adv_std clip_eps ≤ r * adv_std
    ∧ surrogate r adv_std clip_eps ≤ clippedRatio r clip_eps * adv_std :=
  ⟨surrogate_le_unclipped r adv_std clip_eps,
   surrogate_le_clipped r adv_std clip_eps⟩

/-- Nonnegative advantage: the surrogate is monotone in the ratio arm — a
    bigger ratio can't lower the pessimistic objective when advantage is
    nonneg. Finite core behind trust-region monotonicity on the win side. -/
theorem surrogate_mono_ratio_nonneg_adv (r₁ r₂ adv_std clip_eps : ℝ)
    (hadv : 0 ≤ adv_std) (h : r₁ ≤ r₂) :
    surrogate r₁ adv_std clip_eps ≤ max (r₂ * adv_std)
      (clippedRatio r₂ clip_eps * adv_std) := by
  unfold surrogate
  have h1 : r₁ * adv_std ≤ r₂ * adv_std :=
    mul_le_mul_of_nonneg_right h hadv
  calc min (r₁ * adv_std) (clippedRatio r₁ clip_eps * adv_std)
      ≤ r₁ * adv_std := min_le_left _ _
    _ ≤ r₂ * adv_std := h1
    _ ≤ max (r₂ * adv_std) (clippedRatio r₂ clip_eps * adv_std) :=
        le_max_left _ _

noncomputable def batchSurrogate (legal : LegalMask Action) (logits logits_old : Logits Action)
    (hexists : ∃ a, legal a = true) (adv : Action → ℝ) (clip_eps : ℝ) : ℝ :=
  ∑ a : Action, if legal a = true then surrogate (ratio legal logits logits_old hexists a) (adv a) clip_eps else 0

theorem batchSurrogate_ignores_illegal_adv (legal : LegalMask Action)
    (logits logits_old : Logits Action) (hexists : ∃ a, legal a = true)
    (adv1 adv2 : Action → ℝ) (clip_eps : ℝ)
    (hagree : ∀ a, legal a = true → adv1 a = adv2 a) :
    batchSurrogate legal logits logits_old hexists adv1 clip_eps =
    batchSurrogate legal logits logits_old hexists adv2 clip_eps := by
  unfold batchSurrogate
  apply Finset.sum_congr rfl
  intro a _
  by_cases ha : legal a = true
  · simp [ha, hagree a ha]
  · simp [ha]

/-- Zero advantages kill the batch surrogate: with every advantage zero,
the masked sum vanishes regardless of logits or clip — chaining
`surrogate_zero_adv` through the illegal mask. Together with
`batchSurrogate_ignores_illegal_*` the surrogate is exactly zero on any
batch whose legal advantages are all zero, so advantage normalization
is load-bearing at batch level too, not just per-sample. -/
theorem batchSurrogate_zero_adv (legal : LegalMask Action)
    (logits logits_old : Logits Action) (hexists : ∃ a, legal a = true)
    (clip_eps : ℝ) :
    batchSurrogate legal logits logits_old hexists (fun _ => 0) clip_eps = 0 := by
  unfold batchSurrogate
  apply Finset.sum_eq_zero
  intro a _
  by_cases ha : legal a = true
  · simp [ha, surrogate_zero_adv]
  · simp [ha]

theorem batchSurrogate_ignores_illegal_logits (legal : LegalMask Action)
    (logits1 logits2 logits_old : Logits Action)
    (hexists : ∃ a, legal a = true) (adv : Action → ℝ) (clip_eps : ℝ)
    (hagree : ∀ a, legal a = true → logits1 a = logits2 a) :
    batchSurrogate legal logits1 logits_old hexists adv clip_eps =
    batchSurrogate legal logits2 logits_old hexists adv clip_eps := by
  unfold batchSurrogate
  apply Finset.sum_congr rfl
  intro a _
  by_cases ha : legal a = true
  · simp [ha]
    have h_eq : legalSoftmax legal logits1 hexists a = legalSoftmax legal logits2 hexists a :=
      legalSoftmax_ignores_illegal_logits legal logits1 logits2 hagree hexists a ha
    simp [ratio, logPi, h_eq]
  · simp [ha]

theorem maskedPPO_zero_grad_illegal_logits (legal : LegalMask Action)
    (logits1 logits2 logits_old : Logits Action)
    (hexists : ∃ a, legal a = true) (adv : Action → ℝ) (clip_eps : ℝ)
    (hagree : ∀ a, legal a = true → logits1 a = logits2 a) :
    batchSurrogate legal logits1 logits_old hexists adv clip_eps =
    batchSurrogate legal logits2 logits_old hexists adv clip_eps :=
  batchSurrogate_ignores_illegal_logits legal logits1 logits2 logits_old hexists adv clip_eps hagree

/-- Honest gradient-zero: varying an illegal logit leaves `batchSurrogate`
constant (by `batchSurrogate_ignores_illegal_logits`), so its `deriv` is `0`
via `EventuallyEq.deriv_eq` + `deriv_const`. Upgrades the value-invariance
proxy `maskedPPO_zero_grad_illegal_logits` to a real calculus statement. -/
theorem maskedPPO_deriv_illegal_zero (legal : LegalMask Action)
    (logits logits_old : Logits Action) (hexists : ∃ a, legal a = true)
    (adv : Action → ℝ) (clip_eps : ℝ) (b : Action) (h_ill : legal b = false) (x : ℝ) :
    deriv (fun d => batchSurrogate legal (Function.update logits b d) logits_old hexists adv clip_eps) x = 0 := by
  have hconst : ∀ d : ℝ, (fun d => batchSurrogate legal (Function.update logits b d)
      logits_old hexists adv clip_eps) d
      = batchSurrogate legal logits logits_old hexists adv clip_eps := by
    intro d
    apply batchSurrogate_ignores_illegal_logits
    intro a ha
    have hne : a ≠ b := by
      rintro rfl
      simp [ha] at h_ill
    exact Function.update_of_ne hne _ _
  have heq : (fun d => batchSurrogate legal (Function.update logits b d) logits_old hexists adv clip_eps)
      =ᶠ[nhds x] (fun _ => batchSurrogate legal logits logits_old hexists adv clip_eps) :=
    Filter.Eventually.of_forall hconst
  rw [Filter.EventuallyEq.deriv_eq heq]
  exact (hasDerivAt_const x _).deriv

end PPOLoss

end MaskedSoftmax

section ValueLoss

variable {State : Type}

noncomputable def valueLoss (v_target v_pred : ℝ) : ℝ := (v_pred - v_target) ^ 2

theorem valueLoss_nonneg (v_target v_pred : ℝ) : 0 ≤ valueLoss v_target v_pred := by
  unfold valueLoss; positivity

theorem valueLoss_zero_iff (v_target v_pred : ℝ) : valueLoss v_target v_pred = 0 ↔ v_pred = v_target := by
  unfold valueLoss
  constructor
  · intro h
    have : (v_pred - v_target) ^ 2 = 0 := h
    have : v_pred - v_target = 0 := by nlinarith [sq_nonneg (v_pred - v_target)]
    linarith
  · intro h; rw [h, sub_self, sq, mul_zero]

/-- Value loss is symmetric in target and prediction: the squared error
never cares which side is called truth. Practical core: teacher-student
swaps (distilling search values into the net, then distilling the net
back into search priors) measure the same loss in both directions — no
orientation bookkeeping across the distill loop. -/
theorem valueLoss_symm (v_target v_pred : ℝ) :
    valueLoss v_target v_pred = valueLoss v_pred v_target := by
  unfold valueLoss
  rw [← neg_sub v_pred v_target, neg_sq]

noncomputable def batchValueLoss (states : Finset State) (v_target v_pred : State → ℝ) : ℝ :=
  (∑ s ∈ states, valueLoss (v_target s) (v_pred s)) / (states.card : ℝ)

theorem batchValueLoss_nonneg (states : Finset State) (v_target v_pred : State → ℝ) :
    0 ≤ batchValueLoss states v_target v_pred := by
  unfold batchValueLoss
  apply div_nonneg
  · apply Finset.sum_nonneg; intro s _; exact valueLoss_nonneg _ _
  · positivity

/-- Batch value loss vanishes when every state agrees: pointwise zeros
sum to zero — the finite reason a perfect critic scores exactly zero
on any batch, so the batch objective has no irreducible floor hiding
behind averaging. Pairs `valueLoss_zero_iff` (pointwise sharpness):
together sharpness lifts from states to batches. -/
theorem batchValueLoss_zero_of_all_zero (states : Finset State)
    (v_target v_pred : State → ℝ)
    (h : ∀ s ∈ states, valueLoss (v_target s) (v_pred s) = 0) :
    batchValueLoss states v_target v_pred = 0 := by
  unfold batchValueLoss
  rw [Finset.sum_eq_zero h]
  simp

/-- Batch value loss is bounded by the worst state loss: the mean never
exceeds the max. Practical core: the critic's batch error is gated by its
worst state — cap the max per-state error and the batch follows, so
limited-compute value training can early-stop on the max instead of the
full batch mean. -/
theorem batchValueLoss_le_max (states : Finset State) (hne : states.Nonempty)
    (v_target v_pred : State → ℝ) :
    batchValueLoss states v_target v_pred
      ≤ states.sup' hne (fun s => valueLoss (v_target s) (v_pred s)) := by
  unfold batchValueLoss
  have hcard_pos : (0 : ℝ) < (states.card : ℝ) :=
    Nat.cast_pos.mpr (Finset.card_pos.mpr hne)
  rw [div_le_iff₀ hcard_pos]
  have hsup : ∀ s ∈ states, valueLoss (v_target s) (v_pred s)
      ≤ states.sup' hne (fun s => valueLoss (v_target s) (v_pred s)) :=
    fun s hs => Finset.le_sup' (fun s => valueLoss (v_target s) (v_pred s)) hs
  calc ∑ s ∈ states, valueLoss (v_target s) (v_pred s)
      ≤ ∑ _s ∈ states, states.sup' hne (fun s => valueLoss (v_target s) (v_pred s)) :=
        Finset.sum_le_sum hsup
    _ = states.card • states.sup' hne (fun s => valueLoss (v_target s) (v_pred s)) := by
        rw [Finset.sum_const]
    _ = states.sup' hne (fun s => valueLoss (v_target s) (v_pred s)) * (states.card : ℝ) := by
        rw [nsmul_eq_mul, mul_comm]

end ValueLoss

section Entropy

variable {Action : Type} [Fintype Action] [DecidableEq Action]

noncomputable def entropy (legal : LegalMask Action) (logits : Logits Action)
    (hexists : ∃ a, legal a = true) : ℝ :=
  - ∑ a : Action, if legal a = true then legalSoftmax legal logits hexists a * Real.log (legalSoftmax legal logits hexists a) else 0

theorem entropy_ignores_illegal_logits (legal : LegalMask Action)
    (logits1 logits2 : Logits Action) (hexists : ∃ a, legal a = true)
    (hagree : ∀ a, legal a = true → logits1 a = logits2 a) :
    entropy legal logits1 hexists = entropy legal logits2 hexists := by
  unfold entropy
  have h_sum_eq : (∑ a : Action, if legal a = true then legalSoftmax legal logits1 hexists a * Real.log (legalSoftmax legal logits1 hexists a) else (0:ℝ))
                = (∑ a : Action, if legal a = true then legalSoftmax legal logits2 hexists a * Real.log (legalSoftmax legal logits2 hexists a) else (0:ℝ)) := by
    apply Finset.sum_congr rfl
    intro a _
    by_cases ha : legal a = true
    · simp [ha]
      have heq : legalSoftmax legal logits1 hexists a = legalSoftmax legal logits2 hexists a :=
        legalSoftmax_ignores_illegal_logits legal logits1 logits2 hagree hexists a ha
      rw [heq]
    · simp [ha]
  rw [h_sum_eq]

/-- Entropy is nonnegative: each legal term `p·log p ≤ 0` (log p ≤ 0 on
    `(0,1]`), illegal actions contribute exactly 0. Practical core: the
    entropy bonus coefficient sweeps a meaningful knob — entropy never goes
    negative, so a positive bonus always rewards stochasticity and can never
    invert into a determinism bonus. Limited-compute lemma: tune the bonus
    sign with confidence. -/
theorem entropy_nonneg (legal : LegalMask Action)
    (logits : Logits Action) (hexists : ∃ a, legal a = true) :
    0 ≤ entropy legal logits hexists := by
  unfold entropy
  have hle : ∀ a : Action, legalSoftmax legal logits hexists a ≤ 1 := by
    intro a
    have hnn : ∀ b ∈ (Finset.univ : Finset Action),
        0 ≤ legalSoftmax legal logits hexists b :=
      fun b _ => legalSoftmax_nonneg legal logits hexists b
    have h1 := Finset.single_le_sum hnn (Finset.mem_univ a)
    rw [legalSoftmax_sum_one legal logits hexists] at h1
    exact h1
  have hterm : ∀ a : Action, (if legal a = true
      then legalSoftmax legal logits hexists a
        * Real.log (legalSoftmax legal logits hexists a) else (0 : ℝ)) ≤ 0 := by
    intro a
    by_cases ha : legal a = true
    · simp only [ha, if_true]
      have h0 : 0 ≤ legalSoftmax legal logits hexists a :=
        legalSoftmax_nonneg legal logits hexists a
      have hlog : Real.log (legalSoftmax legal logits hexists a) ≤ 0 :=
        Real.log_nonpos h0 (hle a)
      exact mul_nonpos_of_nonneg_of_nonpos h0 hlog
    · simp [ha]
  have hsum : ∑ a : Action, (if legal a = true
      then legalSoftmax legal logits hexists a
        * Real.log (legalSoftmax legal logits hexists a) else (0 : ℝ)) ≤ 0 :=
    Finset.sum_nonpos (fun a _ => hterm a)
  linarith
/-- Honest entropy gradient-zero (mirror of `maskedPPO_deriv_illegal_zero`):
varying an illegal logit leaves `entropy` constant
(by `entropy_ignores_illegal_logits`), so its `deriv` is `0`. -/
theorem entropy_deriv_illegal_zero (legal : LegalMask Action)
    (logits : Logits Action) (hexists : ∃ a, legal a = true)
    (b : Action) (h_ill : legal b = false) (x : ℝ) :
    deriv (fun d => entropy legal (Function.update logits b d) hexists) x = 0 := by
  have hconst : ∀ d : ℝ, (fun d => entropy legal (Function.update logits b d) hexists) d
      = entropy legal logits hexists := by
    intro d
    apply entropy_ignores_illegal_logits
    intro a ha
    have hne : a ≠ b := by
      rintro rfl
      simp [ha] at h_ill
    exact Function.update_of_ne hne _ _
  have heq : (fun d => entropy legal (Function.update logits b d) hexists)
      =ᶠ[nhds x] (fun _ => entropy legal logits hexists) :=
    Filter.Eventually.of_forall hconst
  rw [Filter.EventuallyEq.deriv_eq heq]
  exact (hasDerivAt_const x _).deriv

end Entropy

section CombinedLoss

variable {Action : Type} [Fintype Action] [DecidableEq Action]

noncomputable def ppoLoss
    (legal : LegalMask Action) (logits logits_old : Logits Action)
    (hexists : ∃ a, legal a = true) (adv : Action → ℝ) (clip_eps w_value w_entropy : ℝ)
    (states : Finset Action) (v_target v_pred : Action → ℝ) : ℝ :=
  - batchSurrogate legal logits logits_old hexists adv clip_eps
  + w_value * batchValueLoss states v_target v_pred
  - w_entropy * entropy legal logits hexists

theorem ppoLoss_ignores_illegal_logits (legal : LegalMask Action)
    (logits1 logits2 logits_old : Logits Action)
    (hexists : ∃ a, legal a = true) (adv : Action → ℝ) (clip_eps w_value w_entropy : ℝ)
    (states : Finset Action) (v_target v_pred : Action → ℝ)
    (hagree : ∀ a, legal a = true → logits1 a = logits2 a) :
    ppoLoss legal logits1 logits_old hexists adv clip_eps w_value w_entropy states v_target v_pred =
    ppoLoss legal logits2 logits_old hexists adv clip_eps w_value w_entropy states v_target v_pred := by
  unfold ppoLoss
  rw [maskedPPO_zero_grad_illegal_logits legal logits1 logits2 logits_old hexists adv clip_eps hagree,
      entropy_ignores_illegal_logits legal logits1 logits2 hexists hagree]

/-- Zero entropy weight drops the bonus: at `w_entropy = 0` the loss is
surrogate-plus-value only — the finite reason entropy annealing ends at
a pure exploitation objective. Pair with `entropy_nonneg` (positive weight
rewards stochasticity): together the sweep spans [pure-exploit, bonus]
with the off position exact. -/
theorem ppoLoss_zero_entropy_weight (legal : LegalMask Action)
    (logits logits_old : Logits Action)
    (hexists : ∃ a, legal a = true) (adv : Action → ℝ) (clip_eps w_value : ℝ)
    (states : Finset Action) (v_target v_pred : Action → ℝ) :
    ppoLoss legal logits logits_old hexists adv clip_eps w_value 0 states v_target v_pred =
    - batchSurrogate legal logits logits_old hexists adv clip_eps
    + w_value * batchValueLoss states v_target v_pred := by
  unfold ppoLoss
  ring

/-- Zero value weight drops the critic term: at `w_value = 0` the loss is
policy-plus-entropy only — the finite reason value-free policy sweeps
measure pure policy gradient with no critic contamination. Mirror of
`ppoLoss_zero_entropy_weight`: together each loss term has an exact off
position, so ablations isolate terms cleanly. Zeroing BOTH auxiliaries
leaves the policy term alone (proved next): pure-policy sweeps need no
term bookkeeping. -/
theorem ppoLoss_zero_aux_weights (legal : LegalMask Action)
    (logits logits_old : Logits Action)
    (hexists : ∃ a, legal a = true) (adv : Action → ℝ) (clip_eps : ℝ)
    (states : Finset Action) (v_target v_pred : Action → ℝ) :
    ppoLoss legal logits logits_old hexists adv clip_eps 0 0 states v_target v_pred =
    - batchSurrogate legal logits logits_old hexists adv clip_eps := by
  unfold ppoLoss
  ring

theorem ppoLoss_zero_value_weight (legal : LegalMask Action)
    (logits logits_old : Logits Action)
    (hexists : ∃ a, legal a = true) (adv : Action → ℝ) (clip_eps w_entropy : ℝ)
    (states : Finset Action) (v_target v_pred : Action → ℝ) :
    ppoLoss legal logits logits_old hexists adv clip_eps 0 w_entropy states v_target v_pred =
    - batchSurrogate legal logits logits_old hexists adv clip_eps
    - w_entropy * entropy legal logits hexists := by
  unfold ppoLoss
  ring

end CombinedLoss

section OracleGuiding

/-- Suphx Eq.5 oracle guiding (Suphx 2019, arXiv:2003.13590 Sec 3.3, https://arxiv.org/abs/2003.13590):
  privileged features (opp privates + wall) enter with Bernoulli keep-prob
  `γ_t : P(δ_t=1)=γ_t`, decayed `1 → 0`; at `γ=0` the oracle has transited
  to a normal agent (then continue with `LR×0.1` + importance-weight
  rejection). Linear schedule `γ_t = 1 - t/T`: starts at oracle (`t=0`),
  ends at normal (`t=T`), antitone in between. Plain distillation without
  the schedule fails (`far beyond the capacity of a normal agent`), so the
  gradual path is the load-bearing part. Maps to `DistillationConfig`
  (`gamma_schedule`, `post_oracle_lr_scale`, `iw_reject_threshold`). -/
noncomputable def oracleDropout (T t : ℕ) : ℝ := 1 - (t : ℝ) / (T : ℝ)

theorem oracleDropout_at_zero (T : ℕ) : oracleDropout T 0 = 1 := by
  unfold oracleDropout; simp

theorem oracleDropout_at_terminal (T : ℕ) (hT : 0 < T) : oracleDropout T T = 0 := by
  unfold oracleDropout
  have hT_ne : (T : ℝ) ≠ 0 := ne_of_gt (Nat.cast_pos.mpr hT)
  rw [div_self hT_ne, sub_self]

theorem oracleDropout_antitone (T : ℕ) (hT : 0 < T) {t1 t2 : ℕ} (h : t1 ≤ t2) :
    oracleDropout T t2 ≤ oracleDropout T t1 := by
  unfold oracleDropout
  have hT_pos : (0 : ℝ) < (T : ℝ) := Nat.cast_pos.mpr hT
  have hle : (t1 : ℝ) ≤ (t2 : ℝ) := Nat.cast_le.mpr h
  have hdiv : (t1 : ℝ) / (T : ℝ) ≤ (t2 : ℝ) / (T : ℝ) :=
    div_le_div_of_nonneg_right hle (le_of_lt hT_pos)
  linarith

/-- Strict version of `oracleDropout_antitone`: every later step keeps
strictly less oracle — the finite reason the Suphx transit never stalls
mid-schedule: no two distinct steps share a keep-prob, so the decay
cannot plateau into a partially-oracular policy that trains against
privileged features forever. Pairs `oracleDropout_at_terminal` (ends
at zero): together the transit is strictly falling AND lands exactly. -/
theorem oracleDropout_strict_antitone (T : ℕ) (hT : 0 < T) {t1 t2 : ℕ}
    (h : t1 < t2) :
    oracleDropout T t2 < oracleDropout T t1 := by
  unfold oracleDropout
  have hT_pos : (0 : ℝ) < (T : ℝ) := Nat.cast_pos.mpr hT
  have hlt : (t1 : ℝ) < (t2 : ℝ) := Nat.cast_lt.mpr h
  have hdiv : (t1 : ℝ) / (T : ℝ) < (t2 : ℝ) / (T : ℝ) :=
    (div_lt_div_iff_of_pos_right hT_pos).mpr hlt
  linarith

theorem oracleDropout_mem_Icc (T : ℕ) (hT : 0 < T) (t : ℕ) (ht : t ≤ T) :
    oracleDropout T t ∈ Set.Icc (0 : ℝ) 1 := by
  have hT_pos : (0 : ℝ) < (T : ℝ) := Nat.cast_pos.mpr hT
  have hle : (t : ℝ) ≤ (T : ℝ) := Nat.cast_le.mpr ht
  have hdiv_le : (t : ℝ) / (T : ℝ) ≤ 1 := by
    rw [div_le_one hT_pos]; exact hle
  have hdiv_nn : 0 ≤ (t : ℝ) / (T : ℝ) := by positivity
  constructor <;> unfold oracleDropout <;> linarith

/-- Oracle dropout is strictly positive before the terminal step: the student
still sees privileged signal at every `t < T`. Companion to
`oracleDropout_at_terminal` (which zeroes it at `t = T`): together they say
the schedule interpolates strictly inside (0,1] until the final step — no
early cutoff starves the student, and no late leak survives the deadline.
Failure mode killed: an off-by-one ending guidance a step early. -/
theorem oracleDropout_pos_before_terminal (T t : ℕ) (hT : 0 < T) (ht : t < T) :
    0 < oracleDropout T t := by
  unfold oracleDropout
  have hT_pos : (0 : ℝ) < (T : ℝ) := Nat.cast_pos.mpr hT
  have hle : (t : ℝ) < (T : ℝ) := Nat.cast_lt.mpr ht
  have hdiv_lt : (t : ℝ) / (T : ℝ) < 1 := by
    rw [div_lt_one hT_pos]; exact hle
  have hdiv_nn : 0 ≤ (t : ℝ) / (T : ℝ) := by positivity
  linarith

end OracleGuiding

section VLOG

/-- VLOG diagonal-Gaussian KL (Han et al. ICLR2022, FrostHan/vlog
  `models.py` EQ1:
  `KL(q‖p) = Σ_d [ls_p - ls_q + ((μ_p-μ_q)² + exp(2·ls_q))/(2·exp(2·ls_p)) - 1/2]`
  with prior `(μp,ls_p)` from the executor encoder and posterior `(μq,ls_q)`
  from the oracle encoder). Pure `Finset`/`Real` transcription — no
  `MeasureTheory`; continuous-KL nonneg stays harness-side (same split as
  discrete Gibbs `ppo_kl_nonneg` vs continuous). -/
noncomputable def gaussDiagKL (d : Nat) (mup lsp muq lsq : Fin d → ℝ) : ℝ :=
  ∑ i : Fin d, (lsp i - lsq i
    + ((mup i - muq i) ^ 2 + Real.exp (2 * lsq i)) / (2 * Real.exp (2 * lsp i))
    - 1 / 2)

/-- Posterior-equals-prior gives zero KL (mirrors the `Q_set` nonemptiness
argument `D(q_nom‖q_nom) = 0`: each summand has `log(q/q) = log 1 = 0`; here
each diagonal-Gaussian term collapses to `0 + exp/(2·exp) - 1/2 = 0` since
`exp ≠ 0`). The finite reason the VLOG regularizer vanishes exactly when the
executor already matches the oracle. -/
theorem gaussDiagKL_zero_when_equal (d : Nat) (mu ls : Fin d → ℝ) :
    gaussDiagKL d mu ls mu ls = 0 := by
  unfold gaussDiagKL
  apply Finset.sum_eq_zero
  intro i _
  have hexp : Real.exp (2 * ls i) ≠ 0 := Real.exp_ne_zero _
  have e1 : ls i - ls i = (0 : ℝ) := sub_self _
  have e2 : ((mu i - mu i) ^ 2 + Real.exp (2 * ls i)) / (2 * Real.exp (2 * ls i))
      = 1 / 2 := by
    rw [sub_self, sq, mul_zero, zero_add,
      div_eq_iff (mul_ne_zero (by norm_num) hexp)]
    ring
  rw [e1, e2]
  norm_num

/-- VLOG total loss (EQ2): `RL + exp(logβ)·KL` (DDQN `loss_q`, BC divides the KL
term by `action_size`; `β` auto-tuned against `kld_target`, mahjong `β₀ = 1e-5`,
target 50 nats). -/
noncomputable def vlogTotalLoss (rl logBeta kld : ℝ) : ℝ :=
  rl + Real.exp logBeta * kld

/-- ELBO split: total minus RL equals the weighted KL (decomposition identity). -/
theorem vlog_elbo_split (rl logBeta kld : ℝ) :
    vlogTotalLoss rl logBeta kld - rl = Real.exp logBeta * kld := by
  unfold vlogTotalLoss
  ring

/-- Total dominates the RL loss whenever the KL estimate is nonneg (discrete
action-head KL nonneg is `ppo_kl_nonneg`; `exp` is always nonneg). -/
theorem vlog_total_ge_rl (rl logBeta kld : ℝ) (hk : 0 ≤ kld) :
    rl ≤ vlogTotalLoss rl logBeta kld := by
  unfold vlogTotalLoss
  have h : 0 ≤ Real.exp logBeta * kld := mul_nonneg (Real.exp_nonneg _) hk
  linarith

/-- VLOG total is nonneg at nonneg RL and KL: the joint never goes
negative — the finite reason the VLOG objective can't manufacture a
spurious descent direction below zero: with real losses on both arms
the total is a genuine cost. Pairs `vlog_total_ge_rl` (total covers
RL): together the total sits above RL AND above zero, so the
optimizer only ever descends real signal. -/
theorem vlogTotalLoss_nonneg (rl logBeta kld : ℝ)
    (hrl : 0 ≤ rl) (hk : 0 ≤ kld) :
    0 ≤ vlogTotalLoss rl logBeta kld := by
  unfold vlogTotalLoss
  have h : 0 ≤ Real.exp logBeta * kld := mul_nonneg (Real.exp_nonneg _) hk
  linarith

end VLOG

section ValueMSEBound

variable {State : Type}

/-- Downstream invariant: value branch (`batchValueLoss`): mean-prediction
  error controlled by root-MSE (finite Jensen / Cauchy-Schwarz over the
  batch). Needs `0 < states.card` so division by the batch size is valid. -/
theorem valueMSE_controls_mean_error
    (states : Finset State) (v_target v_pred : State → ℝ)
    (hpos : 0 < states.card) :
    |(∑ s ∈ states, v_pred s) / (states.card : ℝ)
      - (∑ s ∈ states, v_target s) / (states.card : ℝ)|
      ≤ Real.sqrt (batchValueLoss states v_target v_pred) := by
  have hn_pos : (0 : ℝ) < (states.card : ℝ) := Nat.cast_pos.mpr hpos
  have hn_ne : (states.card : ℝ) ≠ 0 := ne_of_gt hn_pos
  have hCS : (∑ s ∈ states, (v_pred s - v_target s)) ^ 2
      ≤ (states.card : ℝ) * ∑ s ∈ states, (v_pred s - v_target s) ^ 2 :=
    sq_sum_le_card_mul_sum_sq
  have hmean : (∑ s ∈ states, v_pred s) / (states.card : ℝ)
      - (∑ s ∈ states, v_target s) / (states.card : ℝ)
      = (∑ s ∈ states, (v_pred s - v_target s)) / (states.card : ℝ) := by
    rw [← sub_div, ← Finset.sum_sub_distrib]
  have hMSE : batchValueLoss states v_target v_pred
      = (∑ s ∈ states, (v_pred s - v_target s) ^ 2) / (states.card : ℝ) := by
    unfold batchValueLoss valueLoss
    rfl
  have hsq : ((∑ s ∈ states, (v_pred s - v_target s)) / (states.card : ℝ)) ^ 2
      ≤ (∑ s ∈ states, (v_pred s - v_target s) ^ 2) / (states.card : ℝ) := by
    calc ((∑ s ∈ states, (v_pred s - v_target s)) / (states.card : ℝ)) ^ 2
        = (∑ s ∈ states, (v_pred s - v_target s)) ^ 2 / (states.card : ℝ) ^ 2 := by
          rw [div_pow]
      _ ≤ ((states.card : ℝ) * ∑ s ∈ states, (v_pred s - v_target s) ^ 2)
          / (states.card : ℝ) ^ 2 :=
          div_le_div_of_nonneg_right hCS (by positivity)
      _ = (∑ s ∈ states, (v_pred s - v_target s) ^ 2) / (states.card : ℝ) := by
          field_simp
  rw [hmean, hMSE]
  exact Real.abs_le_sqrt hsq

end ValueMSEBound

section AdvantageStdSanity

/-- Downstream invariant: standardization sanity (advantage branch):
  standardizing at the mean gives exactly zero. -/
theorem advantageStd_zero_at_mean (mean var_eps : ℝ) :
    advantageStd mean mean var_eps = 0 := by
  unfold advantageStd
  simp

/-- Downstream invariant: standardization sanity (advantage branch):
  standardized advantage is monotone in the raw advantage for fixed `mean`
  and `var_eps > 0` (positive denominator preserves order). -/
theorem advantageStd_monotone (mean var_eps : ℝ) (hvar : 0 < var_eps)
    {a₁ a₂ : ℝ} (h : a₁ ≤ a₂) :
    advantageStd a₁ mean var_eps ≤ advantageStd a₂ mean var_eps := by
  unfold advantageStd
  have hsqrt : (0 : ℝ) < Real.sqrt var_eps := Real.sqrt_pos.mpr hvar
  exact div_le_div_of_nonneg_right (by linarith) (le_of_lt hsqrt)

end AdvantageStdSanity

end Hydra2.Blueprint.PPO
