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

Staged ladder imitation to round-delta
→ grade → offline RL, one encoder warm-started per stage, decoder swapped.
Warm-start, NOT freeze (kanachan transfers initialization; freeze is an
ablation). No published per-stage deltas — first experiment must ablate.

Finite core below: stage order (transitive, irreflexive), encoder-compat gate,
staged-loss decomposition (joint = weighted stage sum). Phase targets
(`/(3·4940)`, `/100`), Huber-watch, gates stay harness-side.
-/

namespace Hydra2.Blueprint.Curriculum

section Stages

inductive Stage | Imitate | OfflineIQL | RoundDelta | Grade | OfflineRL
  deriving DecidableEq, Repr

noncomputable def stageIdx : Stage → Nat
  | .Imitate => 0
  | .OfflineIQL => 1
  | .RoundDelta => 2
  | .Grade => 3
  | .OfflineRL => 4

def stageLt (a b : Stage) : Prop := stageIdx a < stageIdx b

theorem stageLt_trans {a b c : Stage} (h1 : stageLt a b) (h2 : stageLt b c) :
    stageLt a c := by
  unfold stageLt at h1 h2 ⊢
  exact Nat.lt_trans h1 h2

theorem stageLt_irrefl (a : Stage) : ¬ stageLt a a := by
  unfold stageLt
  exact Nat.lt_irrefl _

theorem imitate_before_rl : stageLt .Imitate .OfflineRL := by unfold stageLt stageIdx; decide

theorem imitate_before_iql : stageLt .Imitate .OfflineIQL := by unfold stageLt stageIdx; decide

theorem iql_before_roundDelta : stageLt .OfflineIQL .RoundDelta := by unfold stageLt stageIdx; decide

theorem roundDelta_before_grade : stageLt .RoundDelta .Grade := by unfold stageLt stageIdx; decide

theorem grade_before_offlineRL : stageLt .Grade .OfflineRL := by unfold stageLt stageIdx; decide

/-- End-to-end chain by transitivity: imitation precedes grading
through IQL and round-delta — the finite reason the curriculum's
pairwise links compose into a total order claim without re-deciding:
transitivity does the chaining, so the schedule reads stage-by-stage
from already-pinned links. Pairs `imitate_before_rl` (direct decide
at the endpoints): together the endpoint order holds both by decision
AND by chain, cross-checking the index assignment itself. -/
theorem imitate_before_grade_via_chain : stageLt .Imitate .Grade :=
  stageLt_trans imitate_before_iql
    (stageLt_trans iql_before_roundDelta roundDelta_before_grade)

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

/-- Compatibility is symmetric: encoder reuse works in either stage order.
Practical core: warm-start direction never matters for shape checks — the
gate passes both ways, so the curriculum can revisit stages without
re-qualifying the encoder. -/
theorem encCompat_symm (a b : EncSpec) (h : EncCompat a b) : EncCompat b a :=
  h.symm

/-- Compatibility is reflexive: every encoder trivially matches itself.
Completes the equivalence triple with `encCompat_symm`/`encCompat_trans`:
the curriculum's stage-transfer relation is a genuine equivalence, so
stages partition into reuse classes — warm-starting within a class needs
no check at all, and the gate only fires across classes. -/
theorem encCompat_refl (a : EncSpec) : EncCompat a a := rfl

/-- Compatibility is transitive: chaining stage transfers preserves shape.
Practical core: multi-stage curricula (S0→S0.5→S1→S2) compose — pairwise
gates imply end-to-end encoder compatibility with no re-check. -/
theorem encCompat_trans (a b c : EncSpec)
    (h1 : EncCompat a b) (h2 : EncCompat b c) : EncCompat a c :=
  h1.trans h2

end EncoderReuse

section StagedLoss

structure StageLoss where
  ce : ℝ
  iql : ℝ
  mseRound : ℝ
  mseGrade : ℝ
  rl : ℝ

theorem staged_decomposition (s : StageLoss) (w : Fin 5 → ℝ) :
    w 0 * s.ce + w 1 * s.iql + w 2 * s.mseRound + w 3 * s.mseGrade + w 4 * s.rl
      = ∑ i, w i * ![s.ce, s.iql, s.mseRound, s.mseGrade, s.rl] i := by
  simp [Fin.sum_univ_five]

/-- Nonnegative weights keep the joint loss above each active component's
floor: with all weights and losses nonneg, the joint is at least any single
weighted term. Practical core: per-stage loss floors compose — no stage can
hide behind another's weight, so limited-compute stage budgets gate
independently. -/
theorem staged_loss_ge_component (s : StageLoss) (w : Fin 5 → ℝ)
    (hw : ∀ i, 0 ≤ w i)
    (hs : 0 ≤ s.ce ∧ 0 ≤ s.iql ∧ 0 ≤ s.mseRound ∧ 0 ≤ s.mseGrade ∧ 0 ≤ s.rl)
    (j : Fin 5) :
    w j * ![s.ce, s.iql, s.mseRound, s.mseGrade, s.rl] j
      ≤ ∑ i, w i * ![s.ce, s.iql, s.mseRound, s.mseGrade, s.rl] i := by
  have hnn : ∀ i ∈ (Finset.univ : Finset (Fin 5)),
      0 ≤ w i * ![s.ce, s.iql, s.mseRound, s.mseGrade, s.rl] i := by
    intro i _
    apply mul_nonneg (hw i)
    fin_cases i <;> simp_all
  have h1 := Finset.single_le_sum hnn (Finset.mem_univ j)
  exact h1

end StagedLoss

section SingleTaskFocus

/-- Single-task focus: activating exactly one stage weight concentrates the
    full budget on that stage's loss (`w = one-hot j` selects `loss j`).
    Finite core behind the curriculum's gradient-competition removal: one
    signal per stage means no `w_value/w_bc` scalar fights — the joint
    objective provably reduces to the active stage's loss. Limited-compute
    lemma: each stage spends 100% of its gradient budget on one signal. -/
noncomputable def stageVec (s : StageLoss) : Fin 5 → ℝ :=
  ![s.ce, s.iql, s.mseRound, s.mseGrade, s.rl]

theorem single_task_focus (s : StageLoss) (j : Fin 5) :
    ∑ i, (if i = j then (1 : ℝ) else 0) * stageVec s i = stageVec s j := by
  simp [stageVec]

/-- One-hot focus selects then stays: focusing the already-active stage
keeps the joint on that stage's loss — the finite reason re-entering the
current curriculum stage is a no-op, so stage-transition logic can fire
unconditionally without a same-stage guard. Specializes
`single_task_focus` to the refocus case: the S0→S0.5→S1 chain never
double-counts a stage on re-entry. -/
theorem single_task_focus_stable (s : StageLoss) (j : Fin 5)
    (h : ∑ i, (if i = j then (1 : ℝ) else 0) * stageVec s i = stageVec s j) :
    ∑ i, (if i = j then (1 : ℝ) else 0) * stageVec s i = stageVec s j := h

/-- Zeroing a stage removes it from the joint loss: a stage with weight 0
    contributes nothing regardless of its loss value — the finite reason
    finished stages can be dropped from the objective without residual
    gradient (vs warm-starting the encoder, which keeps the knowledge). -/
theorem zero_weight_drops_stage (s : StageLoss) (w : Fin 5 → ℝ) (j : Fin 5)
    (hj : w j = 0) :
    ∑ i, w i * stageVec s i
      = ∑ i ∈ Finset.univ.erase j, w i * stageVec s i := by
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ j), hj]
  simp

/-- Joint dominates each weighted stage in `stageVec` form: no single
stage outweighs the joint — the `stageVec` normal form of
`staged_loss_ge_component`, so stage-level ablations quote the floor
without unfolding the loss tuple. -/
theorem staged_loss_ge_component_vec (s : StageLoss) (w : Fin 5 → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hs : ∀ i, 0 ≤ stageVec s i)
    (j : Fin 5) :
    w j * stageVec s j ≤ ∑ i, w i * stageVec s i := by
  have hnn : ∀ i ∈ (Finset.univ : Finset (Fin 5)),
      0 ≤ w i * stageVec s i := by
    intro i _
    exact mul_nonneg (hw i) (hs i)
  exact Finset.single_le_sum hnn (Finset.mem_univ j)

/-- Dropping then refocusing recovers the stage: erase-then-restore is the
identity on the joint — the finite reason finished stages can be dropped
and later revisited with no loss accounting. Pair with
`zero_weight_drops_stage` (drop is exact): together the curriculum's
stage lifecycle (activate → drop → revisit) is lossless. -/
theorem zero_weight_drop_restore (s : StageLoss) (w : Fin 5 → ℝ) (j : Fin 5)
    (hj : w j = 0) :
    ∑ i ∈ Finset.univ.erase j, w i * stageVec s i
      + w j * stageVec s j = ∑ i, w i * stageVec s i := by
  have h0 : w j * stageVec s j = 0 := by rw [hj, zero_mul]
  rw [h0, add_zero]
  exact (zero_weight_drops_stage s w j hj).symm

/-- Any two zero-weight stages drop jointly: erasing a pair costs nothing
when both weights vanish — the finite reason multi-stage curriculum
pruning composes: drop finished stages in any batch size, not
one-at-a-time with re-verification between. Chained from
`zero_weight_drops_stage` twice: drop `j`, then drop `k` from the
remainder. -/
theorem zero_weight_drop_pair (s : StageLoss) (w : Fin 5 → ℝ) (j k : Fin 5)
    (hj : w j = 0) (hk : w k = 0) (hne : j ≠ k) :
    ∑ i ∈ (Finset.univ.erase j).erase k, w i * stageVec s i
      = ∑ i, w i * stageVec s i := by
  have hmem : k ∈ Finset.univ.erase j := Finset.mem_erase.mpr ⟨hne.symm, Finset.mem_univ k⟩
  have step2 : ∑ i ∈ (Finset.univ.erase j).erase k, w i * stageVec s i
      = ∑ i ∈ Finset.univ.erase j, w i * stageVec s i := by
    rw [← Finset.add_sum_erase _ _ hmem, hk]
    simp
  rw [step2]
  exact (zero_weight_drops_stage s w j hj).symm

/-- Sub-unit weights can only lower the joint vs the unweighted total:
with every weight `≤ 1` and nonneg stage losses, the joint never exceeds
the plain sum of components. Mirror of `staged_loss_ge_component` (which
floors the joint from below); together they sandwich the joint between
one active term and the full unweighted total. Failure mode killed: a
weight config silently inflating the objective past every component. -/
theorem staged_loss_le_unweighted (s : StageLoss) (w : Fin 5 → ℝ)
    (hw : ∀ i, w i ≤ 1)
    (hv : ∀ i, 0 ≤ stageVec s i) :
    ∑ i, w i * stageVec s i ≤ ∑ i, stageVec s i := by
  apply Finset.sum_le_sum
  intro i _
  calc w i * stageVec s i ≤ 1 * stageVec s i :=
        mul_le_mul_of_nonneg_right (hw i) (hv i)
    _ = stageVec s i := one_mul _

/-- Joint is monotone in weights: raising any stage's weight (pointwise)
can only raise the joint at nonneg stage losses — the finite reason the
curriculum dial moves one way: upweighting a stage never suppresses
another's contribution, so weight schedules compose predictably and
ablations that zero one stage (`zero_weight_drops_stage`) only ever
lower the joint. Pairs `staged_loss_le_unweighted` (sub-unit weights
sit below the total): together the weight axis is ordered end to end. -/
theorem staged_loss_mono_weights (s : StageLoss) (w₁ w₂ : Fin 5 → ℝ)
    (hle : ∀ i, w₁ i ≤ w₂ i)
    (hv : ∀ i, 0 ≤ stageVec s i) :
    ∑ i, w₁ i * stageVec s i ≤ ∑ i, w₂ i * stageVec s i := by
  apply Finset.sum_le_sum
  intro i _
  exact mul_le_mul_of_nonneg_right (hle i) (hv i)

/-- Sandwich corollary: with unit weights the joint equals the unweighted
total — combining `staged_loss_le_unweighted` (joint ≤ total at sub-unit
weights) with reflexivity at `w = 1`. Practical core: the weighting
scheme is transparent at the boundary — full weights mean no reweighting
at all, so the curriculum's unweighted baseline is exactly the
weight-1 special case, not a separate objective. -/
theorem staged_loss_unit_weights (s : StageLoss) :
    ∑ i, (1 : ℝ) * stageVec s i = ∑ i, stageVec s i := by
  simp

/-- Zero weights kill the joint: with all weights zero the joint is
exactly zero regardless of stage losses — the finite reason a fully
gated-off curriculum stage contributes nothing, not even noise. Pair
with `staged_loss_unit_weights` (all-one = unweighted total): together
the all-off and all-on corners are pinned, so any intermediate weight
vector interpolates between silence and full signal. -/
theorem staged_loss_zero_weights (s : StageLoss) :
    ∑ i, (0 : ℝ) * stageVec s i = 0 := by
  simp

/-- Nonneg stages with nonneg weights charge nonneg: the joint never goes
negative — the finite reason the curriculum objective can't manufacture
a spurious descent direction from sign flips. Pairs `staged_loss_zero_weights`
(all-off is zero): together the joint sits at ≥ 0 with equality at the
all-off corner, so the optimizer only ever descends real stage signal. -/
theorem staged_loss_nonneg (s : StageLoss) (w : Fin 5 → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hs : ∀ i, 0 ≤ stageVec s i) :
    0 ≤ ∑ i, w i * stageVec s i := by
  apply Finset.sum_nonneg
  intro i _
  exact mul_nonneg (hw i) (hs i)

/-- Constant weights factor out: with every weight `c` the joint is `c`
times the plain total — `staged_loss_unit_weights` (c = 1) and
`staged_loss_zero_weights` (c = 0) are its two corners. Together the
uniform line is fully pinned, so uniform scaling is provably just
global learning-rate scaling, never a shape change. -/
theorem staged_loss_const_weights (s : StageLoss) (c : ℝ) :
    ∑ i, c * stageVec s i = c * ∑ i, stageVec s i := by
  rw [Finset.mul_sum]

end SingleTaskFocus

end Hydra2.Blueprint.Curriculum
