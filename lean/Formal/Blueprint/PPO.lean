import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Tactic

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false
set_option linter.style.whitespace false

/-! # Hydra2 §20 PPO/ACH — Masked Objectives & Illegal Logit Zero-Grad
Mirrors `IMPLEMENTATION_SPEC.md §20 Masked PPO comparador` and
`ALGORITHM_EXPERIMENT_BLUEPRINT.md` Candidate training objectives.
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

noncomputable def surrogate (r adv_std : ℝ) (clip_eps : ℝ) : ℝ :=
  min (r * adv_std) (clippedRatio r clip_eps * adv_std)

theorem surrogate_le_unclipped (r adv_std clip_eps : ℝ) :
    surrogate r adv_std clip_eps ≤ r * adv_std :=
  min_le_left _ _

theorem surrogate_le_clipped (r adv_std clip_eps : ℝ) :
    surrogate r adv_std clip_eps ≤ clippedRatio r clip_eps * adv_std :=
  min_le_right _ _

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

noncomputable def batchValueLoss (states : Finset State) (v_target v_pred : State → ℝ) : ℝ :=
  (∑ s ∈ states, valueLoss (v_target s) (v_pred s)) / (states.card : ℝ)

theorem batchValueLoss_nonneg (states : Finset State) (v_target v_pred : State → ℝ) :
    0 ≤ batchValueLoss states v_target v_pred := by
  unfold batchValueLoss
  apply div_nonneg
  · apply Finset.sum_nonneg; intro s _; exact valueLoss_nonneg _ _
  · positivity

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

end CombinedLoss

section OracleGuiding

/-- Suphx Eq.5 oracle guiding (SuphxPipe scout, ar5iv 2003.13590 §3.3): privileged features (opp privates + wall) enter with Bernoulli keep-prob `γ_t : P(δ_t=1)=γ_t`, decayed `1 → 0`; at `γ=0` the oracle has transited to a normal agent (then continue with `LR×0.1` + importance-weight rejection). Linear schedule `γ_t = 1 - t/T`: starts at oracle (`t=0`), ends at normal (`t=T`), antitone in between. Plain distillation without the schedule fails (`far beyond the capacity of a normal agent`), so the gradual path is the load-bearing part. Maps to `DistillationConfig` (`gamma_schedule`, `post_oracle_lr_scale`, `iw_reject_threshold`). -/
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

theorem oracleDropout_mem_Icc (T : ℕ) (hT : 0 < T) (t : ℕ) (ht : t ≤ T) :
    oracleDropout T t ∈ Set.Icc (0 : ℝ) 1 := by
  have hT_pos : (0 : ℝ) < (T : ℝ) := Nat.cast_pos.mpr hT
  have hle : (t : ℝ) ≤ (T : ℝ) := Nat.cast_le.mpr ht
  have hdiv_le : (t : ℝ) / (T : ℝ) ≤ 1 := by
    rw [div_le_one hT_pos]; exact hle
  have hdiv_nn : 0 ≤ (t : ℝ) / (T : ℝ) := by positivity
  constructor <;> unfold oracleDropout <;> linarith

end OracleGuiding

section VLOG

/-- VLOG diagonal-Gaussian KL (Han et al. ICLR2022, code-primary
FrostHan/vlog `models.py` EQ1, via IdeaVlog scout:
`KL(q‖p) = Σ_d [ls_p - ls_q + ((μ_p-μ_q)² + exp(2·ls_q))/(2·exp(2·ls_p)) - 1/2]`
with prior `(μp,ls_p)` from the executor encoder and posterior `(μq,ls_q)` from
the oracle encoder). Pure `Finset`/`Real` transcription — no `MeasureTheory`;
continuous-KL nonneg stays harness-side (same split as discrete Gibbs
`ppo_kl_nonneg` vs continuous). -/
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

end VLOG

end Hydra2.Blueprint.PPO
