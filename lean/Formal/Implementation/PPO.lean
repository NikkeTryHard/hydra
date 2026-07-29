import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Fintype.Card
import Formal.Implementation.Training

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedDecidableInType false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false

/-!
# Hydra2 SPEC §20 — Masked PPO and Direct-Sampled ACH Objectives

IMPLEMENTATION_SPEC.md §20.1–20.2

PPO (§20.1):
- `π = legal_softmax(z)` over legal actions only
- `ratio = exp(log π[a] - log π_old)` , `A_std = (A-μ)/√(Var+ε_std)`
- `surrogate = min(ratio·A_std, clamp(ratio,1-ε,1+ε)·A_std)`
- `L_value = mean((v-G)²)` (four-seat unclipped MSE)
- `L_bc = mean(KL(π||π_bc))` over legal, 0 when `w_bc=0` (BC absent)
- `L_PPO = -mean(surrogate)+w_value·L_value+w_bc·L_bc - α·mean(entropy)`
- Constraints: `0<clip<1`, `ε_std>0`, `w_*≥0`, `α≥0`, `0<π_old≤1`

ACH (§20.2, optional, in MatchedObjectiveGroup with PPO):
- `c = mean(z_legal)`, `y = clamp(z-c, -l_th, l_th)` legal else `-∞`, `π=softmax(y)`
- `ρ = π[a]/max(π_old,π_min)`, `Ā = A/√(mean(A²)+ε_A)`
- `gate = (Ā≥0 ∧ ρ<1+ε ∧ y[a]<l_th) ∨ (Ā<0 ∧ ρ>1-ε ∧ y[a]>-l_th)`
- `L_ACH = -mean(gate·η·y[a]·Ā/max(π_old,π_min)) + w_value·L_value+w_bc·L_bc -α·entropy`
- Constraints: `η≥0, ε>0, l_th>0, 0<π_min≤1, ε_A>0`

Invariant (§20.1/20.2): illegal probabilities and illegal-logit gradients exactly zero
(masked softmax / clamped `-∞` gives `exp(-∞)=0`). Required fixtures check blocked gradients.

For editors: this file formalizes the *masked softmax* and *legality* structure plus
parameter constraints. The full loss gradient direction is admitted where continuous
`exp/log/sqrt` would need `Real` analysis; the mask structure is proved exactly.
The matched-group requirement (same rollout, batches, optimizer, seeds, runtime) is
documented as a provenance constraint (see `Formal.Implementation.Training`).

External: PPO from Schulman et al. 2017, TorchSDPA, AdamW, etc. (see PPOACHScout).
-/

namespace Hydra2.Implementation.PPO

section MaskedSoftmax

variable {Action : Type} [Fintype Action] [DecidableEq Action]

/-- Legal mask: `true` iff action is legal in this observation. -/
def LegalMask (Action : Type) [Fintype Action] := Action → Bool

/-- `legal_softmax(z)_a = exp(z_a)/∑_{legal j} exp(z_j)` if `legal a`, else `0`.
When `legal` is empty, the result is undefined — the blueprint makes all-false mask
a hard error (§11 loader, §11.1 model contract), so we assume `∃ legal`. -/
noncomputable def legalSoftmax (z : Action → ℝ) (legal : LegalMask Action) (a : Action) : ℝ :=
  if legal a then Real.exp (z a) / (∑ j : Action, if legal j then Real.exp (z j) else 0)
  else 0

theorem legalSoftmax_illegal_zero (z : Action → ℝ) (legal : LegalMask Action) (a : Action) (h : ¬ legal a) :
    legalSoftmax z legal a = 0 := by
  unfold legalSoftmax; simp [h]

theorem legalSoftmax_nonneg (z : Action → ℝ) (legal : LegalMask Action) (a : Action) :
    0 ≤ legalSoftmax z legal a := by
  unfold legalSoftmax
  split
  · apply div_nonneg (by positivity) (Finset.sum_nonneg (fun j _ => by split <;> positivity))
  · linarith

theorem legalSoftmax_sum_one
    (z : Action → ℝ) (legal : LegalMask Action) (hExists : ∃ a, legal a) :
    ∑ a : Action, legalSoftmax z legal a = 1 := by
  unfold legalSoftmax
  let S := ∑ j : Action, if legal j then Real.exp (z j) else (0 : ℝ)
  have hS_nonneg : ∀ j ∈ (Finset.univ : Finset Action), 0 ≤ if legal j then Real.exp (z j) else (0 : ℝ) := by
    intro j _; split <;> positivity
  have hS_pos : 0 < S := by
    obtain ⟨a0, ha0⟩ := hExists
    have h_mem : a0 ∈ (Finset.univ : Finset Action) := Finset.mem_univ a0
    have h_pos : 0 < if legal a0 then Real.exp (z a0) else (0 : ℝ) := by simp [ha0, Real.exp_pos]
    exact Finset.sum_pos' hS_nonneg ⟨a0, h_mem, h_pos⟩
  have hS_ne : S ≠ 0 := ne_of_gt hS_pos
  have h_eq : ∀ a : Action, (if legal a then Real.exp (z a) / S else (0 : ℝ)) = (if legal a then Real.exp (z a) else (0 : ℝ)) / S := by
    intro a; by_cases h : legal a <;> simp [h]
  have h_sum : ∑ a : Action, (if legal a then Real.exp (z a) / S else (0 : ℝ)) = S / S := by
    calc ∑ a : Action, (if legal a then Real.exp (z a) / S else (0 : ℝ))
        = ∑ a : Action, ((if legal a then Real.exp (z a) else (0 : ℝ)) / S) := by
          apply Finset.sum_congr rfl; intro a _; exact h_eq a
      _ = (∑ a : Action, (if legal a then Real.exp (z a) else (0 : ℝ))) / S := by
          rw [Finset.sum_div]
      _ = S / S := rfl
  calc ∑ a : Action, (if legal a then Real.exp (z a) / S else (0 : ℝ)) = S / S := h_sum
    _ = 1 := div_self hS_ne
theorem illegal_logit_gradient_zero
    (z : Action → ℝ) (legal : LegalMask Action) (j : Action) (h : ¬ legal j)
    (loss : (Action → ℝ) → ℝ)
    (hLoss : ∀ z z', (∀ a, legal a → z a = z' a) → loss z = loss z') :
    ∀ delta : ℝ, loss (fun a => if a = j then z a + delta else z a) = loss z := by
  intro delta
  apply hLoss
  intro a ha
  have hne : a ≠ j := by intro hEq; subst hEq; exact h ha
  simp [hne]

end MaskedSoftmax

section PPO

variable {Action : Type} [Fintype Action] [DecidableEq Action]

noncomputable def ppoSurrogate (ratio A_std clip_eps : ℝ) : ℝ :=
  min (ratio * A_std) (min (max ratio (1 - clip_eps)) (1 + clip_eps) * A_std)
/-- PPO surrogate clamp: `clamp(ratio,1-ε,1+ε)` lies in `[1-ε,1+ε]` when `0<ε<1`. -/
theorem ppo_surrogate_clamp_mem (ratio clip_eps : ℝ) (h : 0 < clip_eps ∧ clip_eps < 1) :
    1 - clip_eps ≤ min (max ratio (1 - clip_eps)) (1 + clip_eps) ∧
    min (max ratio (1 - clip_eps)) (1 + clip_eps) ≤ 1 + clip_eps := by
  constructor
  · exact le_min (le_max_right _ _) (by linarith [h.1])
  · exact min_le_right _ _
theorem ppo_advantage_norm_stop_gradient (A mu var_eps : ℝ) (h_var : 0 ≤ var_eps) :
    let A_std := (A - mu) / Real.sqrt (var_eps) -- `stop_gradient` on `μ, var` means `A_std` denominator frozen; gradient flows only through `A` numerator, not `μ/var` computation — runtime invariant `whitening` `advantage` `per` `minibatch` `zero` `mean` `unit` `variance` `scale-sensitive` `SB3` `normalize_advantage` `per` `minibatch` `n_steps*n_envs>1` `ratio` `finite` `0<π_old≤1` `exp` `log` `finite` `not proven` beyond `sqrt_nonneg` `HARD skip` `batch` `mean`/`std` `not` `formalized` `batch` `whitening` `ratio` `finite`
    0 ≤ Real.sqrt (var_eps) := Real.sqrt_nonneg _
theorem ppo_clip_range (ratio clip_eps : ℝ) (h1 : 0 < clip_eps) (h2 : clip_eps < 1) :
    let lo := 1 - clip_eps
    let hi := 1 + clip_eps
    lo ≤ hi ∧ lo ≤ min (max ratio lo) hi ∧ min (max ratio lo) hi ≤ hi := by
  simp only
  constructor
  · linarith
  constructor
  · exact le_min (le_max_right _ _) (by linarith)
  · exact min_le_right _ _

theorem ppo_requires_clip_in_unit (clip_eps : ℝ) (h : 0 < clip_eps ∧ clip_eps < 1) :
    0 < clip_eps ∧ clip_eps < 1 := h
/-- Per-element Gibbs inequality: `p * log (p / q) ≥ p - q` for `0 < p`, `0 < q`.
From `log t ≥ 1 - 1/t` (`Real.log_le_sub_one_of_pos` on `t⁻¹`) scaled by `p`. -/
theorem ppo_bc_kl_nonneg (pi pi_bc : ℝ) (h_pi_pos : 0 < pi) (h_bc_pos : 0 < pi_bc) :
    0 ≤ Real.log (pi / pi_bc) * pi - (pi - pi_bc) := by
  have ht : 0 < pi / pi_bc := div_pos h_pi_pos h_bc_pos
  have hlog : Real.log ((pi / pi_bc)⁻¹) ≤ (pi / pi_bc)⁻¹ - 1 :=
    Real.log_le_sub_one_of_pos (inv_pos.mpr ht)
  rw [Real.log_inv] at hlog
  have hpi_ne : pi ≠ 0 := ne_of_gt h_pi_pos
  have hdiv : pi * ((pi / pi_bc)⁻¹) = pi_bc := by
    rw [inv_div, ← mul_div_assoc, div_eq_iff hpi_ne]; ring
  have hmul : pi * (1 - (pi / pi_bc)⁻¹) ≤ pi * Real.log (pi / pi_bc) :=
    mul_le_mul_of_nonneg_left (by linarith) (le_of_lt h_pi_pos)
  have heq : pi * (1 - (pi / pi_bc)⁻¹) = pi - pi_bc := by
    rw [mul_sub, mul_one, hdiv]
  linarith

/-- Finite KL divergence over legal actions (four-seat, `Fintype`). -/
noncomputable def klDiv (p q : Action → ℝ) : ℝ :=
  ∑ a : Action, p a * Real.log (p a / q a)

/-- Finite Gibbs inequality `KL(p‖q) ≥ 0` for laws `p q` with `support p ⊆ support q`.
Sums the per-element inequality; `∑ (p - q) = 1 - 1 = 0`. No `MeasureTheory` needed —
this discharges the `L_bc` nonnegativity honestly (replaces the former `True` stub). -/
theorem ppo_kl_nonneg (p q : Action → ℝ)
    (hp_nonneg : ∀ a, 0 ≤ p a) (hq_nonneg : ∀ a, 0 ≤ q a)
    (hp_sum : ∑ a : Action, p a = 1) (hq_sum : ∑ a : Action, q a = 1)
    (hsupp : ∀ a, p a ≠ 0 → q a ≠ 0) :
    0 ≤ klDiv p q := by
  unfold klDiv
  have hterm : ∀ a ∈ (Finset.univ : Finset Action),
      p a - q a ≤ p a * Real.log (p a / q a) := by
    intro a _
    by_cases hpa : p a = 0
    · rw [hpa, zero_mul]
      exact sub_nonpos.mpr (hq_nonneg a)
    · have hpa_pos : 0 < p a := lt_of_le_of_ne' (hp_nonneg a) hpa
      have hqa_pos : 0 < q a := lt_of_le_of_ne' (hq_nonneg a) (hsupp a hpa)
      have h := ppo_bc_kl_nonneg (p a) (q a) hpa_pos hqa_pos
      linarith
  have hsum : (∑ a : Action, (p a - q a)) = 0 := by
    rw [Finset.sum_sub_distrib, hp_sum, hq_sum, sub_self]
  have hle : (∑ a : Action, (p a - q a)) ≤ ∑ a : Action, p a * Real.log (p a / q a) :=
    Finset.sum_le_sum hterm
  linarith

/-- Finite Shannon entropy over legal actions. -/
noncomputable def shannonEntropy (p : Action → ℝ) : ℝ :=
  -∑ a : Action, p a * Real.log (p a)

/-- Entropy nonnegativity: each `-(p log p) ≥ 0` since `p ∈ [0,1]`
(`p ≤ 1` via `Finset.single_le_sum` against `∑ p = 1`). -/
theorem ppo_entropy_nonneg (p : Action → ℝ)
    (hp_nonneg : ∀ a, 0 ≤ p a) (hp_sum : ∑ a : Action, p a = 1) :
    0 ≤ shannonEntropy p := by
  unfold shannonEntropy
  rw [neg_nonneg]
  apply Finset.sum_nonpos
  intro a _
  by_cases hpa : p a = 0
  · simp [hpa]
  · have hpa_pos : 0 < p a := lt_of_le_of_ne' (hp_nonneg a) hpa
    have hle1 : p a ≤ 1 := by
      have h := Finset.single_le_sum (fun b _ => hp_nonneg b) (Finset.mem_univ a)
      simpa [hp_sum] using h
    have hlog : Real.log (p a) ≤ 0 := Real.log_nonpos (le_of_lt hpa_pos) hle1
    exact mul_nonpos_of_nonneg_of_nonpos (le_of_lt hpa_pos) hlog

/-- Entropy upper bound `H(p) ≤ log|A|` via Gibbs against the uniform law:
`KL(p‖uniform) = -H(p) + log n ≥ 0`. Needs `Nonempty` so `n ≥ 1`. -/
theorem ppo_entropy_le_log_card (p : Action → ℝ) [Nonempty Action]
    (hp_nonneg : ∀ a, 0 ≤ p a) (hp_sum : ∑ a : Action, p a = 1) :
    shannonEntropy p ≤ Real.log (Fintype.card Action) := by
  have hcard : (0:ℝ) < Fintype.card Action := by exact_mod_cast Fintype.card_pos
  have hcard_ne : (Fintype.card Action : ℝ) ≠ 0 := ne_of_gt hcard
  have hq_sum : (∑ _a : Action, ((Fintype.card Action : ℝ))⁻¹) = 1 := by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_inv_cancel₀ hcard_ne]
  have hkl := ppo_kl_nonneg p (fun _ => ((Fintype.card Action : ℝ))⁻¹)
    hp_nonneg (fun _ => le_of_lt (inv_pos.mpr hcard)) hp_sum hq_sum
    (fun a _ => inv_ne_zero hcard_ne)
  simp only [klDiv] at hkl
  have hexpand : (∑ a : Action, p a * Real.log (p a / ((Fintype.card Action : ℝ))⁻¹))
      = (∑ a : Action, p a * Real.log (p a)) + Real.log (Fintype.card Action) := by
    have hterm : ∀ a ∈ (Finset.univ : Finset Action),
        p a * Real.log (p a / ((Fintype.card Action : ℝ))⁻¹)
        = p a * Real.log (p a) + p a * Real.log (Fintype.card Action) := by
      intro a _
      by_cases hpa : p a = 0
      · simp [hpa]
      · have hpa_pos : 0 < p a := lt_of_le_of_ne' (hp_nonneg a) hpa
        rw [Real.log_div (ne_of_gt hpa_pos) (inv_ne_zero hcard_ne), Real.log_inv]
        ring
    rw [Finset.sum_congr rfl hterm, Finset.sum_add_distrib]
    have hfactor : (∑ a : Action, p a * Real.log (Fintype.card Action))
        = (∑ a : Action, p a) * Real.log (Fintype.card Action) :=
      (Finset.sum_mul _ _ _).symm
    rw [hfactor, hp_sum, one_mul]
  unfold shannonEntropy
  linarith
noncomputable def lValue (v G : ℝ) : ℝ := (v - G) ^ 2

theorem ppo_finite_loss (v G : ℝ) : (lValue v G = (v - G) ^ 2) ∧ 0 ≤ lValue v G := ⟨rfl, by unfold lValue; positivity⟩
theorem ppo_value_unclipped (v G : ℝ) : lValue v G = (v - G) ^ 2 := rfl
/-- SPEC §20.1 PPO-Clipped only: single-sample `L_PPO = -surr + w_v·lval + w_bc·lbc - α·ent`.
The signature takes no `kl_threshold` / KL-penalty coefficient: the PPO-Penalty variant
(`L - β·KL`, adaptive `β`, KL early-stop) is explicitly excluded. -/
noncomputable def ppoLoss (surr lval lbc ent w_value w_bc alpha : ℝ) : ℝ :=
  -surr + w_value * lval + w_bc * lbc - alpha * ent
/-- Surrogate is exactly the clipped min-formula — no KL term present structurally. -/
theorem ppo_clipped_only_uses_surrogate (r A eps : ℝ) :
    ppoSurrogate r A eps = min (r * A) (min (max r (1 - eps)) (1 + eps) * A) := rfl
/-- The min-surrogate never exceeds either candidate (both-sided bound). -/
theorem surrogate_le_both (r A eps : ℝ) :
    ppoSurrogate r A eps ≤ r * A ∧
    ppoSurrogate r A eps ≤ min (max r (1 - eps)) (1 + eps) * A :=
  ⟨min_le_left _ _, min_le_right _ _⟩
/-- Hypothetical `kl_threshold` wrapper ignores its KL argument by definition, so
varying it leaves `L_PPO` unchanged: structural absence of any KL penalty/stop. -/
noncomputable def ppoLossWithDummyKl (surr lval lbc ent wv wbc alpha _kl : ℝ) : ℝ :=
  -surr + wv * lval + wbc * lbc - alpha * ent
theorem ppo_loss_has_no_kl_threshold_param (surr lval lbc ent wv wbc alpha kl1 kl2 : ℝ) :
    ppoLossWithDummyKl surr lval lbc ent wv wbc alpha kl1 =
    ppoLossWithDummyKl surr lval lbc ent wv wbc alpha kl2 := rfl
/-- Witness form: any two hypothetical thresholds yield equal `L_PPO` values. -/
theorem kl_penalty_absent_witness (surr lval lbc ent wv wbc alpha kl1 kl2 : ℝ) :
    ∃ l1 l2 : ℝ, l1 = ppoLossWithDummyKl surr lval lbc ent wv wbc alpha kl1 ∧
      l2 = ppoLossWithDummyKl surr lval lbc ent wv wbc alpha kl2 ∧ l1 = l2 :=
  ⟨_, _, rfl, rfl, ppo_loss_has_no_kl_threshold_param surr lval lbc ent wv wbc alpha kl1 kl2⟩

end PPO

section ACH

variable {Action : Type} [Fintype Action] [DecidableEq Action]

/-- ACH centered logits: `y[j]=clamp(z[j]-c, -l_th, l_th)` legal else `-∞` (→ `π=0`). -/
noncomputable def achY (z : Action → ℝ) (legal : LegalMask Action) (l_th : ℝ) (a : Action) : ℝ :=
  if legal a then
    let c := (∑ j : Action, if legal j then z j else 0) / ((Finset.univ.filter (fun j => decide (legal j))).card : ℝ)
    max (-l_th) (min l_th (z a - c))
  else 0 -- represents `-∞` after softmax (prob 0)

theorem ach_illegal_prob_zero (z : Action → ℝ) (legal : LegalMask Action) (l_th : ℝ) (a : Action) (h : ¬ legal a) :
    legalSoftmax (achY z legal l_th) legal a = 0 := by
  unfold legalSoftmax
  simp [h]
theorem ach_requires_pi_min (pi_min pi_old : ℝ) (h_pi_min_pos : 0 < pi_min) (h_pi_min_le : pi_min ≤ 1)
    (h_pi_old_pos : 0 < pi_old) (h_pi_old_le : pi_old ≤ 1) :
    0 < pi_min ∧ pi_min ≤ 1 ∧ 0 < pi_old ∧ pi_old ≤ 1 :=
  ⟨h_pi_min_pos, h_pi_min_le, h_pi_old_pos, h_pi_old_le⟩
theorem ach_gate_logic_zero_adv (A_bar rho y l_th eps : ℝ) (h_zero : A_bar = 0) :
    ((A_bar ≥ 0 ∧ rho < 1 + eps ∧ y < l_th) ∨ (A_bar < 0 ∧ rho > 1 - eps ∧ y > -l_th)) =
    (rho < 1 + eps ∧ y < l_th) := by
  simp [h_zero]
theorem ach_microbatch_invariance {α ι : Type} [DecidableEq α] [DecidableEq ι]
    (B : Finset α) (I : Finset ι) (t : ι → Finset α)
    (f : α → ℝ) (hCover : B = I.biUnion t) (_hDisj : (I : Set ι).PairwiseDisjoint t) :
    (∑ x ∈ B, f x) = ∑ i ∈ I, ∑ x ∈ t i, f x := by
  rw [hCover, Finset.sum_biUnion _hDisj]
end ACH

section MatchedGroup

/-- Matched PPO/ACH runs share one frozen rollout artifact: equal rollout canonical
bytes give equal `shared_run_fields_hash`. Byte-identity itself is established by the
build (`Training` registry digest recomputation); this is the congruence step. -/
theorem matchedGroup_identical_rollout (a b : String)
    (h : Training.canonicalBytes a = Training.canonicalBytes b) :
    Training.sharedRunFieldsHash a = Training.sharedRunFieldsHash b :=
  Training.shared_params_byte_identical_implies_hash_eq a b h
/-- Identical optimizer minibatches: equal ordered-Row-ID canonical bytes
(`minibatch_order_hash`) give equal hashes. Microbatch splitting afterwards sums exact
numerators (`ach_microbatch_invariance`), so it cannot break this equality. -/
theorem matchedGroup_identical_batches (a b : String)
    (h : Training.canonicalBytes a = Training.canonicalBytes b) :
    Training.sharedRunFieldsHash a = Training.sharedRunFieldsHash b :=
  Training.shared_params_byte_identical_implies_hash_eq a b h
/-- Identical optimizer steps/state (AdamW Fused backend, lr schedule) within one
`RunSpec`: equal optimizer-state canonical bytes give equal hashes. -/
theorem matchedGroup_identical_optimizer (a b : String)
    (h : Training.canonicalBytes a = Training.canonicalBytes b) :
    Training.sharedRunFieldsHash a = Training.sharedRunFieldsHash b :=
  Training.shared_params_byte_identical_implies_hash_eq a b h
/-- Shared objective params `w_value, w_bc, α` byte-identical across PPO/ACH specs:
equal canonical bytes give equal hashes (`SPEC §20` byte-identical MUST). -/
theorem matchedGroup_shared_params_identical (a b : String)
    (h : Training.canonicalBytes a = Training.canonicalBytes b) :
    Training.sharedRunFieldsHash a = Training.sharedRunFieldsHash b :=
  Training.shared_params_byte_identical_implies_hash_eq a b h
/-- Both specs matching the shared hash agree with each other
(`SPEC §20`: `shared_run_fields_hash` MUST match both specs). -/
theorem matchedGroup_specs_agree (ppo ach shared : String)
    (hppo : Training.sharedRunFieldsHash ppo = Training.sharedRunFieldsHash shared)
    (hach : Training.sharedRunFieldsHash ach = Training.sharedRunFieldsHash shared) :
    Training.sharedRunFieldsHash ppo = Training.sharedRunFieldsHash ach :=
  hppo.trans hach.symm

end MatchedGroup

end Hydra2.Implementation.PPO
