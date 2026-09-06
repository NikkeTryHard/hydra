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
# Hydra2 Acquisition Scores (EI duality, one-step VoI, BE trade-off)

BPR Rosman et al. 2016 (ar5iv 1505.00284) §3 (acquisition), §3.1.1 (EI),
§3.1.3 (KG/VoI). Bayes-Lab 2022: finite type library + packet signals
imply finite sums throughout. Gaussian CDF (Φ) EI closed form and MC KG
integration stay harness-side; here only finite-sum scores + order algebra.
-/

namespace Hydra2.Blueprint.Acquisition

section AcqScores

variable {Theta Pi : Type} [Fintype Theta] [Fintype Pi]

/-- Finite-sum EI score: posterior-weighted cost-to-go above baseline. -/
def eiScore (beta : Theta → ℝ) (F : Pi → Theta → ℝ → ℝ) (Ubar : ℝ) (p : Pi) : ℝ :=
  ∑ t, beta t * F p t Ubar

/-- Minimizer of the score maximizes EI: `EI = C - score` reverses order. -/
theorem ei_duality {beta : Theta → ℝ} {F : Pi → Theta → ℝ → ℝ} {Ubar C : ℝ}
    {EI : Pi → ℝ} {p q : Pi}
    (hC1 : EI p = C - eiScore beta F Ubar p)
    (hC2 : EI q = C - eiScore beta F Ubar q)
    (hle : eiScore beta F Ubar p ≤ eiScore beta F Ubar q) :
    EI q ≤ EI p := by
  linarith

variable {Sigma : Type} [Fintype Sigma]

/-- One-step VoI ≥ 0: posterior max `V` dominates any prior max `M` under
weights `w` (KG one-step gain, Jensen-free; harness computes `V` by MC). -/
theorem voi_nonneg {w V : Sigma → ℝ} {M : ℝ}
    (hw : ∀ s, 0 ≤ w s) (hM : ∀ s, M ≤ V s)
    (hsum : ∑ s, w s = 1) : M ≤ ∑ s, w s * V s := by
  calc M = M * ∑ s, w s := by rw [hsum, mul_one]
    _ = ∑ s, M * w s := by rw [Finset.mul_sum]
    _ ≤ ∑ s, w s * V s := by
        apply Finset.sum_le_sum
        intro s _
        have h := mul_le_mul_of_nonneg_right (hM s) (hw s)
        calc M * w s ≤ V s * w s := h
          _ = w s * V s := by ring

/-- BE trade-off `Ũ − κH`: no existence claim; argmax lives harness-side. -/
def beScore (util ent : Pi → ℝ) (kappa : ℝ) (p : Pi) : ℝ :=
  util p - kappa * ent p

/-- Zero exploration weight recovers pure exploitation (`κ = 0` picks by `Ũ`
alone — the greedy baseline BE generalizes). -/
theorem be_kappa_zero (util ent : Pi → ℝ) (p : Pi) :
    beScore util ent 0 p = util p := by
  unfold beScore
  ring

/-- BE is antitone in `κ` at fixed policy when entropy is nonneg: more
exploration weight can only lower the exploitative score (the knob BE sweeps
for pure-probe epochs). -/
theorem be_antitone_kappa (util ent : Pi → ℝ) (p : Pi)
    (hent : 0 ≤ ent p) {k1 k2 : ℝ} (h : k1 ≤ k2) :
    beScore util ent k2 p ≤ beScore util ent k1 p := by
  unfold beScore
  have hmul : k1 * ent p ≤ k2 * ent p :=
    mul_le_mul_of_nonneg_right h hent
  linarith

end AcqScores

section HorizonKG
variable {Pi : Type} [Fintype Pi]

/-- KG horizon rule (BPR §3.1.3 eq6-7): `Ũ(π) + (K−t)·ν` — at horizon end
(`K = t`) the probe premium vanishes and the score is pure exploitation.
Finite identity only; VoI estimation stays harness-side. -/
noncomputable def kgHorizonScore (util voi : Pi → ℝ) (K t : ℕ) (p : Pi) : ℝ :=
  util p + ((K : ℝ) - (t : ℝ)) * voi p

theorem kg_horizon_terminal (util voi : Pi → ℝ) (t : ℕ) (p : Pi) :
    kgHorizonScore util voi t t p = util p := by
  unfold kgHorizonScore
  simp

end HorizonKG

end Hydra2.Blueprint.Acquisition
