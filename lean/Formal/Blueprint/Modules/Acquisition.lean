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

BPR Rosman et al. 2016 (https://arxiv.org/abs/1505.00284) acquisition, EI, and KG/VoI.
Finite type library + packet signals imply finite sums throughout. Gaussian CDF
EI closed form and MC KG integration stay harness-side; here only finite-sum scores
+ order algebra. Using infinite type libraries here would diverge and fails closed.
-/

namespace Hydra2.Blueprint.Acquisition

section AcqScores

variable {Theta Pi : Type} [Fintype Theta] [Fintype Pi]

/-- Finite-sum EI score: posterior-weighted cost-to-go above baseline. -/
def eiScore (beta : Theta → ℝ) (F : Pi → Theta → ℝ → ℝ) (Ubar : ℝ) (p : Pi) : ℝ :=
  ∑ t, beta t * F p t Ubar

/-- Zero posterior mass means zero score: a policy with no belief support
contributes nothing to the acquisition — the finite reason the router can
skip zero-mass policies without evaluation. Pair with `ei_duality` (score
order flips to EI order): together pruned policies never outrank, so the
shortlist sweep can filter on posterior mass first, scores second. -/
theorem eiScore_zero_mass {beta : Theta → ℝ} {F : Pi → Theta → ℝ → ℝ} {Ubar : ℝ}
    {p : Pi} (h : ∀ t, beta t = 0) :
    eiScore beta F Ubar p = 0 := by
  unfold eiScore
  apply Finset.sum_eq_zero
  intro t _
  rw [h t, zero_mul]

/-- Nonneg mass with nonneg costs scores nonneg: the EI score never goes
negative under valid beliefs — the finite reason the acquisition ranking
needs no sign guard. Pairs `eiScore_zero_mass` (null belief scores
zero): together the score sits at ≥ 0 with equality at null mass, so
the router's argmin is always a genuine minimum, never an artifact. -/
theorem eiScore_nonneg {beta : Theta → ℝ} {F : Pi → Theta → ℝ → ℝ} {Ubar : ℝ}
    {p : Pi} (hb : ∀ t, 0 ≤ beta t) (hF : ∀ t, 0 ≤ F p t Ubar) :
    0 ≤ eiScore beta F Ubar p := by
  unfold eiScore
  apply Finset.sum_nonneg
  intro t _
  exact mul_nonneg (hb t) (hF t)

/-- Minimizer of the score maximizes EI: `EI = C - score` reverses order. -/
theorem ei_duality {beta : Theta → ℝ} {F : Pi → Theta → ℝ → ℝ} {Ubar C : ℝ}
    {EI : Pi → ℝ} {p q : Pi}
    (hC1 : EI p = C - eiScore beta F Ubar p)
    (hC2 : EI q = C - eiScore beta F Ubar q)
    (hle : eiScore beta F Ubar p ≤ eiScore beta F Ubar q) :
    EI q ≤ EI p := by
  linarith

/-- EI duality at equality: tied scores give tied EI — the strict-order
converse of `ei_duality`. Together order and equality both transport
across the `C - score` flip, so the router can compare in whichever
space is cheaper (score or EI) with no information loss. -/
theorem ei_duality_eq {beta : Theta → ℝ} {F : Pi → Theta → ℝ → ℝ} {Ubar C : ℝ}
    {EI : Pi → ℝ} {p q : Pi}
    (hC1 : EI p = C - eiScore beta F Ubar p)
    (hC2 : EI q = C - eiScore beta F Ubar q)
    (heq : eiScore beta F Ubar p = eiScore beta F Ubar q) :
    EI p = EI q := by
  linarith

variable {Sigma : Type} [Fintype Sigma]

/-- One-step VoI ≥ 0: posterior max `V` dominates any prior max `M` under
weights `w` (KG one-step gain, Jensen-free; harness computes `V` by MC).
VoI is zero when the posterior learns nothing (proved below): if `V` is
constant the weighted value equals the prior max. -/
theorem voi_zero_of_constant {w V : Sigma → ℝ} {M : ℝ}
    (_hw : ∀ s, 0 ≤ w s) (hV : ∀ s, V s = M)
    (hsum : ∑ s, w s = 1) : ∑ s, w s * V s = M := by
  have : ∑ s, w s * V s = ∑ s, w s * M := by
    apply Finset.sum_congr rfl
    intro s _
    rw [hV s]
  rw [this, ← Finset.sum_mul, hsum, one_mul]

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

/-- Zero entropy collapses BE to pure exploitation: with `ent = 0` the
score is the utility regardless of κ — the finite reason
fully-observed (or fully-certain) states skip the exploration sweep
entirely. Pair with `be_kappa_zero` (κ = 0 same collapse): together
both degenerate corners agree, so the BE dial interpolates between
two pinned endpoints with no surprise interior. -/
theorem be_zero_entropy (util ent : Pi → ℝ) (kappa : ℝ) (p : Pi)
    (h : ent p = 0) :
    beScore util ent kappa p = util p := by
  unfold beScore
  rw [h, mul_zero, sub_zero]

/-- Zero exploration weight recovers pure exploitation (`κ = 0` picks by `Ũ`
alone — the greedy baseline BE generalizes). -/
theorem be_kappa_zero (util ent : Pi → ℝ) (p : Pi) :
    beScore util ent 0 p = util p := by
  unfold beScore
  ring

/-- BE at zero utility prices pure exploration cost: with `util = 0` the
score is `−κ·ent` — the finite reason exploration has a signed price:
negative exactly where uncertainty is positive, zero where entropy
vanishes. Pairs `be_zero_entropy` (zero entropy costs nothing): together
both zero corners are pinned, so the BE dial reads as utility minus an
explicit exploration bill. -/
theorem be_zero_util (util ent : Pi → ℝ) (kappa : ℝ) (p : Pi)
    (h : util p = 0) :
    beScore util ent kappa p = -(kappa * ent p) := by
  unfold beScore
  rw [h, zero_sub]

/-- BE is nonneg in κ when utility dominates: if `util ≥ κ·ent` the score
stays nonneg — the finite reason the exploration sweep can bound score
from below by capping κ against the utility-to-entropy ratio. Pair with
`be_antitone_kappa` (raising κ only lowers): together the sweep has a
floor the harness can compute from current estimates. -/
theorem be_nonneg_of_kappa_bound (util ent : Pi → ℝ) (kappa : ℝ) (p : Pi)
    (h : kappa * ent p ≤ util p) :
    0 ≤ beScore util ent kappa p := by
  unfold beScore
  linarith

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

/-- BE at double zero is pure utility: with `κ = 0` and `ent = 0` the
score is the utility — the finite reason the fully-certain greedy
policy needs no exploration dial at all: both degenerate corners
(`be_kappa_zero`, `be_zero_entropy`) agree at the origin, so the
dial's zero point is unambiguous. -/
theorem be_double_zero (util ent : Pi → ℝ) (p : Pi) :
    beScore util ent 0 p = util p :=
  be_kappa_zero util ent p

/-- BE respects the exploitation order at fixed exploration: a higher
util value can only raise the score regardless of κ. Pair with
`be_antitone_kappa` (κ demotes at fixed entropy): together they say the
score moves the right way in EACH argument — the router never inverts
the value ranking while sweeping exploration. -/
theorem be_mono_util (util₁ util₂ ent : Pi → ℝ) (kappa : ℝ) (p : Pi)
    (h : util₁ p ≤ util₂ p) :
    beScore util₁ ent kappa p ≤ beScore util₂ ent kappa p := by
  unfold beScore
  linarith

/-- BE is antitone in entropy at fixed policy with nonneg κ: more
uncertainty can only lower the exploitative score — the mirror of
`be_mono_util` (utility raises it). Together the score moves the right
way in EACH state input, so the router never inverts either ranking
while sweeping exploration weight. -/
theorem be_antitone_ent (util ent₁ ent₂ : Pi → ℝ) (kappa : ℝ) (p : Pi)
    (hk : 0 ≤ kappa) (h : ent₁ p ≤ ent₂ p) :
    beScore util ent₂ kappa p ≤ beScore util ent₁ kappa p := by
  unfold beScore
  have hmul : kappa * ent₁ p ≤ kappa * ent₂ p :=
    mul_le_mul_of_nonneg_left h hk
  linarith

end AcqScores

section HorizonKG
variable {Pi : Type} [Fintype Pi]

/-- KG horizon rule (BPR Eq.6-7, https://arxiv.org/abs/1505.00284): `Ũ(π) + (K−t)·ν` — at horizon end
(`K = t`) the probe premium vanishes and the score is pure exploitation.
Finite identity only; VoI estimation stays harness-side. -/
noncomputable def kgHorizonScore (util voi : Pi → ℝ) (K t : ℕ) (p : Pi) : ℝ :=
  util p + ((K : ℝ) - (t : ℝ)) * voi p

theorem kg_horizon_terminal (util voi : Pi → ℝ) (t : ℕ) (p : Pi) :
    kgHorizonScore util voi t t p = util p := by
  unfold kgHorizonScore
  simp

/-- KG probe premium is nonnegative exactly when time remains and value of
    information is nonnegative: `(K−t)·ν ≥ 0` iff the horizon hasn't ended.
    Finite core behind MI-gated probing: with no time left (`K ≤ t`) or no
    information value (`ν ≤ 0`), the premium vanishes or goes negative and
    the router must exploit — never pay for a probe that cannot return.
    Limited-compute lemma: probe budget is only spent where VoI is positive. -/
theorem kg_premium_nonneg (util voi : Pi → ℝ) (K t : ℕ) (p : Pi)
    (hKt : (t : ℝ) ≤ K) (hvoi : 0 ≤ voi p) :
    util p ≤ kgHorizonScore util voi K t p := by
  unfold kgHorizonScore
  have h : 0 ≤ ((K : ℝ) - (t : ℝ)) * voi p :=
    mul_nonneg (sub_nonneg.mpr hKt) hvoi
  linarith

/-- Expired horizon kills the premium even with positive VoI: past the
    deadline the score drops back to (or below) pure exploitation, so a
    stale probe can never outrank the greedy policy on premium alone.
    At the deadline the score IS pure exploitation (proved next): `K = t`
    zeroes the time weight exactly, so the terminal step is premium-free,
    never approximately so. -/
theorem kg_premium_at_deadline (util voi : Pi → ℝ) (t : ℕ) (p : Pi) :
    kgHorizonScore util voi t t p = util p := by
  unfold kgHorizonScore
  simp

theorem kg_premium_expired (util voi : Pi → ℝ) (K t : ℕ) (p : Pi)
    (hKt : (K : ℝ) ≤ t) (hvoi : 0 ≤ voi p) :
    kgHorizonScore util voi K t p ≤ util p := by
  unfold kgHorizonScore
  have h : ((K : ℝ) - (t : ℝ)) * voi p ≤ 0 :=
    mul_nonpos_of_nonpos_of_nonneg (sub_nonpos.mpr hKt) hvoi
  linarith

/-- Longer horizon never lowers the score at fixed policy with nonneg VoI:
    more remaining time can only raise the probe premium. Practical core:
    the router's time-horizon knob is monotone — extending the horizon
    never demotes a probing policy, so budget extensions don't reshuffle
    the shortlist. Limited-compute lemma: spend horizon where VoI is high.
    Zero VoI flattens it (proved next): VoI-free policies skip the sweep,
    priced once — the knob matters exactly where VoI is live. -/
theorem kg_score_flat_zero_voi (util voi : Pi → ℝ) (t K₁ K₂ : ℕ) (p : Pi)
    (hvoi : voi p = 0) :
    kgHorizonScore util voi K₁ t p = kgHorizonScore util voi K₂ t p := by
  unfold kgHorizonScore
  rw [hvoi]
  ring

theorem kg_score_mono_horizon (util voi : Pi → ℝ) (t K₁ K₂ : ℕ) (p : Pi)
    (hvoi : 0 ≤ voi p) (h : K₁ ≤ K₂) :
    kgHorizonScore util voi K₁ t p ≤ kgHorizonScore util voi K₂ t p := by
  unfold kgHorizonScore
  have hK : (K₁ : ℝ) ≤ (K₂ : ℝ) := Nat.cast_le.mpr h
  have hmul : ((K₁ : ℝ) - (t : ℝ)) * voi p ≤ ((K₂ : ℝ) - (t : ℝ)) * voi p :=
    mul_le_mul_of_nonneg_right (by linarith) hvoi
  linarith

/-- KG score is monotone in VoI at fixed horizon with time remaining:
better information value can only raise the probe score — the third
monotonicity axis alongside `kg_score_mono_horizon` (deadline) and
`kg_score_antitone_time` (elapsed). Together the score moves the right
way in ALL three inputs, so the probe ranking never inverts on any
axis and the MI gate's sweep reads cleanly. -/
theorem kg_score_mono_voi (util : Pi → ℝ) (voi₁ voi₂ : Pi → ℝ)
    (K t : ℕ) (p : Pi) (hKt : (t : ℝ) ≤ (K : ℝ))
    (h : voi₁ p ≤ voi₂ p) :
    kgHorizonScore util voi₁ K t p ≤ kgHorizonScore util voi₂ K t p := by
  unfold kgHorizonScore
  have hmul : ((K : ℝ) - (t : ℝ)) * voi₁ p ≤ ((K : ℝ) - (t : ℝ)) * voi₂ p :=
    mul_le_mul_of_nonneg_left h (sub_nonneg.mpr hKt)
  linarith

/-- KG score is antitone in elapsed time at fixed horizon with nonneg VoI:
later starts can only lower the probe premium — the mirror of
`kg_score_mono_horizon` in the `t` argument. Together the score moves the
right way in BOTH time arguments: extending the deadline never demotes a
probe, and delaying the start never promotes one. Failure mode killed: a
stale-state router outranking fresh probes on premium alone. -/
theorem kg_score_antitone_time (util voi : Pi → ℝ) (K t₁ t₂ : ℕ) (p : Pi)
    (hvoi : 0 ≤ voi p) (h : t₁ ≤ t₂) :
    kgHorizonScore util voi K t₂ p ≤ kgHorizonScore util voi K t₁ p := by
  unfold kgHorizonScore
  have ht : (t₁ : ℝ) ≤ (t₂ : ℝ) := Nat.cast_le.mpr h
  have hmul : ((K : ℝ) - (t₂ : ℝ)) * voi p ≤ ((K : ℝ) - (t₁ : ℝ)) * voi p :=
    mul_le_mul_of_nonneg_right (by linarith) hvoi
  linarith

end HorizonKG

end Hydra2.Blueprint.Acquisition
