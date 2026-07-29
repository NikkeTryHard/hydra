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

Mirrors `ideas/cql-offline-rl.md` (Mortal `train.py` + Kumar et al. 2020):
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

noncomputable def cqlPenalty (legal : LegalMask Action) (Q : Action → ℝ)
    (dataMean : ℝ) : ℝ :=
  maskedLogSumExp legal Q - dataMean

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

/-- Scaled penalty stays nonneg (`α ≥ 0`, e.g. Mortal `min_q_weight` offline-only). -/
theorem cql_scaled_nonneg (legal : LegalMask Action) (Q : Action → ℝ)
    (S : Finset Action) (hne : S.Nonempty) (hleg : ∀ a ∈ S, legal a = true)
    (α : ℝ) (hα : 0 ≤ α) :
    0 ≤ α * cqlPenalty legal Q ((∑ a ∈ S, Q a) / (S.card : ℝ)) := by
  unfold cqlPenalty
  apply mul_nonneg hα
  have h := cql_penalty_nonneg legal Q S hne hleg
  linarith

end MaskedLSE

end Hydra2.Blueprint.CQL
