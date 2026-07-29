import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedDecidableInType false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false

/-! # Hydra2 curriculum fine-tuning with encoder-reuse (kanachan)

Mirrors `ideas/curriculum-finetuning.md`: staged ladder imitation → round-delta
→ grade → offline RL, one encoder warm-started per stage, decoder swapped.
Warm-start, NOT freeze (kanachan transfers initialization; freeze is an
ablation). No published per-stage deltas — first experiment must ablate.

Finite core below: stage order (transitive, irreflexive), encoder-compat gate,
staged-loss decomposition (joint = weighted stage sum). Phase targets
(`/(3·4940)`, `/100`), Huber-watch, gates stay harness-side.
-/

namespace Hydra2.Blueprint.Curriculum

section Stages

inductive Stage | Imitate | RoundDelta | Grade | OfflineRL
  deriving DecidableEq, Repr

noncomputable def stageIdx : Stage → Nat
  | .Imitate => 0
  | .RoundDelta => 1
  | .Grade => 2
  | .OfflineRL => 3

def stageLt (a b : Stage) : Prop := stageIdx a < stageIdx b

theorem stageLt_trans {a b c : Stage} (h1 : stageLt a b) (h2 : stageLt b c) :
    stageLt a c := by
  unfold stageLt at h1 h2 ⊢
  exact Nat.lt_trans h1 h2

theorem stageLt_irrefl (a : Stage) : ¬ stageLt a a := by
  unfold stageLt
  exact Nat.lt_irrefl _

theorem imitate_before_rl : stageLt .Imitate .OfflineRL := by unfold stageLt stageIdx; decide

theorem roundDelta_before_grade : stageLt .RoundDelta .Grade := by unfold stageLt stageIdx; decide

theorem grade_before_offlineRL : stageLt .Grade .OfflineRL := by unfold stageLt stageIdx; decide

end Stages

section EncoderReuse

structure EncSpec where
  dim : Nat
  heads : Nat
  layers : Nat

def EncCompat (a b : EncSpec) : Prop := a = b

theorem reuse_requires_compat (a b : EncSpec) (h : EncCompat a b) :
    a.dim = b.dim ∧ a.heads = b.heads ∧ a.layers = b.layers := by
  obtain ⟨rfl⟩ := h
  exact ⟨rfl, rfl, rfl⟩

end EncoderReuse

section StagedLoss

structure StageLoss where
  ce : ℝ
  mseRound : ℝ
  mseGrade : ℝ
  rl : ℝ

theorem staged_decomposition (s : StageLoss) (w : Fin 4 → ℝ) :
    w 0 * s.ce + w 1 * s.mseRound + w 2 * s.mseGrade + w 3 * s.rl
      = ∑ i, w i * ![s.ce, s.mseRound, s.mseGrade, s.rl] i := by
  simp [Fin.sum_univ_four]

end StagedLoss

end Hydra2.Blueprint.Curriculum
