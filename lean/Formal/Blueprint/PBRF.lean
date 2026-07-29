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
# Hydra2 §10 PBRF Core
-/

namespace Hydra2.Blueprint.PBRF

section PBRFCore

variable {World Packet : Type} [Fintype World] [DecidableEq World] [Fintype Packet] [DecidableEq Packet]

structure KernelEntry (World Packet : Type) [DecidableEq World] [DecidableEq Packet] where
  succ : World
  packet : Packet
  prob : ℝ
  nonneg : 0 ≤ prob

structure ParentPop (World : Type) where
  worlds : List World
  size_pos : 0 < worlds.length

noncomputable def N (pop : ParentPop World) : ℝ := (pop.worlds.length : ℝ)

noncomputable def gammaHat
    (pop : ParentPop World)
    (kernelEntries : World → Finset (KernelEntry World Packet))
    (e : Packet) (g : World → ℝ) : ℝ :=
  (1 / N pop) * (∑ i : Fin pop.worlds.length, ∑ entry ∈ kernelEntries (pop.worlds.get i), if entry.packet = e then entry.prob * g entry.succ else 0)

noncomputable def Zhat (pop : ParentPop World) (kernelEntries : World → Finset (KernelEntry World Packet)) (e : Packet) : ℝ :=
  gammaHat pop kernelEntries e (fun _ => 1)

noncomputable def childHat
    (pop : ParentPop World) (kernelEntries : World → Finset (KernelEntry World Packet)) (e : Packet) (g : World → ℝ) (_he : Zhat pop kernelEntries e ≠ 0) : ℝ :=
  gammaHat pop kernelEntries e g / Zhat pop kernelEntries e

def Exhaustive (pop : ParentPop World) (kernelEntries : World → Finset (KernelEntry World Packet)) : Prop :=
  ∀ x ∈ pop.worlds, (∑ entry ∈ kernelEntries x, entry.prob) = 1

theorem gammaHat_nonneg (pop : ParentPop World) (kernelEntries : World → Finset (KernelEntry World Packet)) (e : Packet) (g : World → ℝ) (hg : ∀ x', 0 ≤ g x') : 0 ≤ gammaHat pop kernelEntries e g := by
  unfold gammaHat N
  apply mul_nonneg
  · apply div_nonneg (by norm_num) (by positivity)
  · apply Finset.sum_nonneg
    intro i _
    apply Finset.sum_nonneg
    intro entry _
    split
    · exact mul_nonneg (entry.nonneg) (hg _)
    · linarith

theorem Zhat_nonneg (pop : ParentPop World) (kernelEntries : World → Finset (KernelEntry World Packet)) (e : Packet) : 0 ≤ Zhat pop kernelEntries e := by
  unfold Zhat
  exact gammaHat_nonneg pop kernelEntries e _ (fun _ => by norm_num)

theorem Zhat_partition
    (pop : ParentPop World) (kernelEntries : World → Finset (KernelEntry World Packet))
    (hExh : Exhaustive pop kernelEntries) :
    ∑ e : Packet, Zhat pop kernelEntries e = 1 := by
  have hN_pos : (0 : ℝ) < (pop.worlds.length : ℝ) := by exact_mod_cast pop.size_pos
  have hN_ne : (pop.worlds.length : ℝ) ≠ 0 := ne_of_gt hN_pos
  have h_mul_one : ∀ (e : Packet) (i : Fin pop.worlds.length) (entry : KernelEntry World Packet),
      (if entry.packet = e then entry.prob * (fun _ : World => (1 : ℝ)) entry.succ else (0 : ℝ)) = if entry.packet = e then entry.prob else 0 := by
    intro e i entry
    by_cases h : entry.packet = e
    · rw [if_pos h, if_pos h]
      ring
    · rw [if_neg h, if_neg h]
  have h_inner_eq : ∀ e : Packet,
      (∑ i : Fin pop.worlds.length, ∑ entry ∈ kernelEntries (pop.worlds.get i), if entry.packet = e then entry.prob * (fun _ : World => (1 : ℝ)) entry.succ else (0 : ℝ))
      = (∑ i : Fin pop.worlds.length, ∑ entry ∈ kernelEntries (pop.worlds.get i), (if entry.packet = e then entry.prob else (0 : ℝ))) := by
    intro e
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro entry _
    exact h_mul_one e i entry
  have h_packet_single : ∀ (entry : KernelEntry World Packet),
      (∑ e : Packet, (if entry.packet = e then entry.prob else (0 : ℝ))) = entry.prob := by
    intro entry
    have h1 : (∑ e : Packet, (if entry.packet = e then entry.prob else (0 : ℝ))) = (∑ e : Packet, (if e = entry.packet then entry.prob else (0 : ℝ))) := by
      apply Finset.sum_congr rfl
      intro e _
      simp only [eq_comm]
    rw [h1, Finset.sum_ite_eq' Finset.univ entry.packet (fun _ => entry.prob)]
    simp
  calc ∑ e : Packet, Zhat pop kernelEntries e
      = (∑ e : Packet, (1 / (pop.worlds.length : ℝ)) * ∑ i : Fin pop.worlds.length, ∑ entry ∈ kernelEntries (pop.worlds.get i), (if entry.packet = e then entry.prob * (fun _ : World => (1 : ℝ)) entry.succ else (0 : ℝ))) := by
        unfold Zhat gammaHat N
        rfl
      _ = (∑ e : Packet, (1 / (pop.worlds.length : ℝ)) * ∑ i : Fin pop.worlds.length, ∑ entry ∈ kernelEntries (pop.worlds.get i), (if entry.packet = e then entry.prob else (0 : ℝ))) := by
        apply Finset.sum_congr rfl
        intro e _
        apply congrArg (HMul.hMul (1 / (pop.worlds.length : ℝ)))
        exact h_inner_eq e
      _ = ((1 / (pop.worlds.length : ℝ)) * ∑ e : Packet, ∑ i : Fin pop.worlds.length, ∑ entry ∈ kernelEntries (pop.worlds.get i), (if entry.packet = e then entry.prob else (0 : ℝ))) := by
        rw [← Finset.mul_sum]
      _ = ((1 / (pop.worlds.length : ℝ)) * ∑ i : Fin pop.worlds.length, ∑ e : Packet, ∑ entry ∈ kernelEntries (pop.worlds.get i), (if entry.packet = e then entry.prob else (0 : ℝ))) := by
        apply congrArg (HMul.hMul (1 / (pop.worlds.length : ℝ)))
        rw [Finset.sum_comm]
      _ = ((1 / (pop.worlds.length : ℝ)) * ∑ i : Fin pop.worlds.length, ∑ entry ∈ kernelEntries (pop.worlds.get i), ∑ e : Packet, (if entry.packet = e then entry.prob else (0 : ℝ))) := by
        apply congrArg (HMul.hMul (1 / (pop.worlds.length : ℝ)))
        apply Finset.sum_congr rfl
        intro i _
        rw [Finset.sum_comm]
      _ = ((1 / (pop.worlds.length : ℝ)) * ∑ i : Fin pop.worlds.length, ∑ entry ∈ kernelEntries (pop.worlds.get i), entry.prob) := by
        apply congrArg (HMul.hMul (1 / (pop.worlds.length : ℝ)))
        apply Finset.sum_congr rfl
        intro i _
        apply Finset.sum_congr rfl
        intro entry _
        exact h_packet_single entry
      _ = ((1 / (pop.worlds.length : ℝ)) * ∑ _i : Fin pop.worlds.length, (1 : ℝ)) := by
        apply congrArg (HMul.hMul (1 / (pop.worlds.length : ℝ)))
        apply Finset.sum_congr rfl
        intro i _
        exact hExh (pop.worlds.get i) (List.get_mem pop.worlds i)
      _ = ((1 / (pop.worlds.length : ℝ)) * (pop.worlds.length : ℝ)) := by
        apply congrArg (HMul.hMul (1 / (pop.worlds.length : ℝ)))
        have h1 : (Finset.univ : Finset (Fin pop.worlds.length)).card = pop.worlds.length := by
          rw [Finset.card_univ, Fintype.card_fin]
        have h2 : (∑ _i : Fin pop.worlds.length, (1 : ℝ)) = ((Finset.univ : Finset (Fin pop.worlds.length)).card : ℝ) := by
          calc (∑ _i : Fin pop.worlds.length, (1 : ℝ)) = ∑ _i ∈ (Finset.univ : Finset (Fin pop.worlds.length)), (1 : ℝ) := rfl
            _ = (((Finset.univ : Finset (Fin pop.worlds.length)).card : ℕ) : ℝ) := by
              rw [Finset.sum_const, nsmul_eq_mul, mul_one]
        rw [h2, h1]
      _ = 1 := by
        rw [one_div, inv_mul_cancel₀ hN_ne]

theorem childHat_normalized
    (pop : ParentPop World) (kernelEntries : World → Finset (KernelEntry World Packet)) (e : Packet) (he : Zhat pop kernelEntries e ≠ 0) :
    childHat pop kernelEntries e (fun _ => 1) he = 1 := by
  unfold childHat Zhat at *
  exact div_self he

/-- `childHat` is a convex combination of `g`-values over the packet-`e`
entries, hence bounded above by any uniform bound `C` (upper half of the
ratio-estimator property; replaces the former `IsRatioEstimator := True`
placeholder with the first genuine ratio-estimator bound). -/
theorem childHat_le_sup
    (pop : ParentPop World) (kernelEntries : World → Finset (KernelEntry World Packet))
    (e : Packet) (g : World → ℝ) (C : ℝ) (hg : ∀ x, g x ≤ C)
    (he : Zhat pop kernelEntries e ≠ 0) (hZ : 0 < Zhat pop kernelEntries e) :
    childHat pop kernelEntries e g he ≤ C := by
  unfold childHat
  rw [div_le_iff₀ hZ]
  have hNN : (0 : ℝ) ≤ 1 / N pop := by
    unfold N
    apply div_nonneg (by norm_num)
    exact_mod_cast Nat.zero_le _
  have hterm : ∀ (_i : Fin pop.worlds.length) (entry : KernelEntry World Packet),
      (if entry.packet = e then entry.prob * g entry.succ else (0 : ℝ))
      ≤ C * (if entry.packet = e then entry.prob else (0 : ℝ)) := by
    intro _ entry
    by_cases h : entry.packet = e
    · simp only [h, if_true]
      calc entry.prob * g entry.succ ≤ entry.prob * C :=
            mul_le_mul_of_nonneg_left (hg _) entry.nonneg
        _ = C * entry.prob := by ring
    · simp [h]
  have hS : (∑ i : Fin pop.worlds.length, ∑ entry ∈ kernelEntries (pop.worlds.get i),
        (if entry.packet = e then entry.prob * g entry.succ else (0 : ℝ)))
      ≤ C * (∑ i : Fin pop.worlds.length, ∑ entry ∈ kernelEntries (pop.worlds.get i),
        (if entry.packet = e then entry.prob else (0 : ℝ))) := by
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i _
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro entry _
    exact hterm i entry
  have hZZ : Zhat pop kernelEntries e = gammaHat pop kernelEntries e (fun _ => 1) := rfl
  rw [hZZ]
  unfold gammaHat
  simp only [mul_one]
  have hC := mul_le_mul_of_nonneg_left hS hNN
  have hring : (1 / N pop) * (C * (∑ i : Fin pop.worlds.length,
      ∑ entry ∈ kernelEntries (pop.worlds.get i),
      (if entry.packet = e then entry.prob else (0 : ℝ))))
      = C * ((1 / N pop) * (∑ i : Fin pop.worlds.length,
      ∑ entry ∈ kernelEntries (pop.worlds.get i),
      (if entry.packet = e then entry.prob else (0 : ℝ)))) := by ring
  rw [hring] at hC
  exact hC

/-- Lower half of the convex-combination bound (mirror of `childHat_le_sup`):
`childHat` is bounded below by any uniform lower bound `c`. Together they
complete the two-sided ratio-estimator bound. -/
theorem childHat_ge_inf
    (pop : ParentPop World) (kernelEntries : World → Finset (KernelEntry World Packet))
    (e : Packet) (g : World → ℝ) (c : ℝ) (hg : ∀ x, c ≤ g x)
    (he : Zhat pop kernelEntries e ≠ 0) (hZ : 0 < Zhat pop kernelEntries e) :
    c ≤ childHat pop kernelEntries e g he := by
  unfold childHat
  rw [le_div_iff₀ hZ]
  have hNN : (0 : ℝ) ≤ 1 / N pop := by
    unfold N
    apply div_nonneg (by norm_num)
    exact_mod_cast Nat.zero_le _
  have hterm : ∀ (_i : Fin pop.worlds.length) (entry : KernelEntry World Packet),
      c * (if entry.packet = e then entry.prob else (0 : ℝ))
      ≤ (if entry.packet = e then entry.prob * g entry.succ else (0 : ℝ)) := by
    intro _ entry
    by_cases h : entry.packet = e
    · simp only [h, if_true]
      calc c * entry.prob = entry.prob * c := by ring
        _ ≤ entry.prob * g entry.succ :=
            mul_le_mul_of_nonneg_left (hg _) entry.nonneg
    · simp [h]
  have hS : c * (∑ i : Fin pop.worlds.length, ∑ entry ∈ kernelEntries (pop.worlds.get i),
        (if entry.packet = e then entry.prob else (0 : ℝ)))
      ≤ (∑ i : Fin pop.worlds.length, ∑ entry ∈ kernelEntries (pop.worlds.get i),
        (if entry.packet = e then entry.prob * g entry.succ else (0 : ℝ))) := by
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i _
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro entry _
    exact hterm i entry
  have hZZ : Zhat pop kernelEntries e = gammaHat pop kernelEntries e (fun _ => 1) := rfl
  rw [hZZ]
  unfold gammaHat
  simp only [mul_one]
  have hC := mul_le_mul_of_nonneg_left hS hNN
  have hring : (1 / N pop) * (c * (∑ i : Fin pop.worlds.length,
      ∑ entry ∈ kernelEntries (pop.worlds.get i),
      (if entry.packet = e then entry.prob else (0 : ℝ))))
      = c * ((1 / N pop) * (∑ i : Fin pop.worlds.length,
      ∑ entry ∈ kernelEntries (pop.worlds.get i),
      (if entry.packet = e then entry.prob else (0 : ℝ)))) := by ring
  rw [hring] at hC
  exact hC

/-- `childHat = gammaHat / Zhat` by definition — the ratio form. -/
theorem childHat_eq_div
    (pop : ParentPop World) (kernelEntries : World → Finset (KernelEntry World Packet)) (e : Packet) (g : World → ℝ) (he : Zhat pop kernelEntries e ≠ 0) :
    childHat pop kernelEntries e g he = gammaHat pop kernelEntries e g / Zhat pop kernelEntries e := rfl

/-- Jensen for `1/X` (convex on `>0`): `E[1/X] ≠ 1/E[X]` generally, so
normalized `η̂ = γ̂/γ̂(1)` is biased `O(1/N)`. Concrete `X∈{1,3}` uniform:
`E[X]=2`, `1/E[X]=1/2` but `E[1/X]=2/3`. -/
theorem childHat_bias_via_jensen :
    ∃ (n : Nat) (hn : 0 < n) (X : Fin n → ℝ) (h_pos : ∀ i, 0 < X i),
      (1 / (n : ℝ)) * ∑ i : Fin n, (1 / X i) ≠ 1 / ((1 / (n : ℝ)) * ∑ i : Fin n, X i) := by
  refine ⟨2, by omega, fun i => if i.val = 0 then (1 : ℝ) else 3, fun i => ?_, ?_⟩
  · fin_cases i <;> simp
  · simp only [Fin.sum_univ_two]
    norm_num
noncomputable def ESS (weights : Finset ℝ) (h_sum_one : ∑ w ∈ weights, w = 1) (h_nonneg : ∀ w ∈ weights, 0 ≤ w) : ℝ :=
  1 / (∑ w ∈ weights, w ^ 2)

theorem ESS_range (weights : Finset ℝ) (h_sum_one : ∑ w ∈ weights, w = 1) (h_nonneg : ∀ w ∈ weights, 0 ≤ w) (hne : weights.Nonempty) :
    1 ≤ ESS weights h_sum_one h_nonneg ∧ ESS weights h_sum_one h_nonneg ≤ (weights.card : ℝ) := by
  unfold ESS
  let S := ∑ w ∈ weights, w ^ 2
  have hS_def : S = ∑ w ∈ weights, w ^ 2 := rfl
  have hcard_pos_nat : 0 < weights.card := Finset.card_pos.mpr hne
  have hcard_pos : (0 : ℝ) < (weights.card : ℝ) := Nat.cast_pos.mpr hcard_pos_nat
  have hcard_ne : (weights.card : ℝ) ≠ 0 := ne_of_gt hcard_pos
  -- S > 0
  have hS_pos : 0 < S := by
    have h_exists_pos : ∃ w ∈ weights, 0 < w := by
      by_contra h
      push_neg at h
      have h_all_zero : ∀ w ∈ weights, w = 0 := by
        intro w hw
        have h_le := h w hw
        have h_nn := h_nonneg w hw
        linarith
      have h_sum_zero : ∑ w ∈ weights, w = 0 := by
        apply Finset.sum_eq_zero
        intro w hw; exact h_all_zero w hw
      linarith
    obtain ⟨w0, hw0_mem, hw0_pos⟩ := h_exists_pos
    have h_w0_sq_pos : 0 < w0 ^ 2 := sq_pos_of_pos hw0_pos
    have h_sum_nonneg : ∀ w ∈ weights, 0 ≤ w ^ 2 := by intro w _; exact sq_nonneg _
    have h_sum_pos : 0 < ∑ w ∈ weights, w ^ 2 := by
      apply Finset.sum_pos' h_sum_nonneg
      exact ⟨w0, hw0_mem, h_w0_sq_pos⟩
    exact h_sum_pos
  have hS_ne : S ≠ 0 := ne_of_gt hS_pos
  -- each w ≤ 1
  have h_w_le_one : ∀ w ∈ weights, w ≤ 1 := by
    intro w hw
    have h_sum_erase : ∑ v ∈ weights.erase w, v + w = ∑ v ∈ weights, v := by
      exact Finset.sum_erase_add _ _ hw
    have h_sum_erase_nonneg : 0 ≤ ∑ v ∈ weights.erase w, v := by
      apply Finset.sum_nonneg; intro v hv
      have hv_mem : v ∈ weights := Finset.mem_of_mem_erase hv
      exact h_nonneg v hv_mem
    linarith
  have h_w_sq_le_w : ∀ w ∈ weights, w ^ 2 ≤ w := by
    intro w hw
    have hw_nn := h_nonneg w hw
    have hw_le := h_w_le_one w hw
    nlinarith
  have hS_le_one : S ≤ 1 := by
    calc S = ∑ w ∈ weights, w ^ 2 := rfl
      _ ≤ ∑ w ∈ weights, w := Finset.sum_le_sum (fun w hw => h_w_sq_le_w w hw)
      _ = 1 := h_sum_one
  -- lower bound S ≥ 1/card via nonnegativity of ∑(w - μ)^2
  let μ : ℝ := 1 / (weights.card : ℝ)
  have h_sum_mu_sq : ∑ _w ∈ weights, μ ^ 2 = (weights.card : ℝ) * μ ^ 2 := by
    rw [Finset.sum_const, nsmul_eq_mul]
  have h_sum_2mu_w : ∑ w ∈ weights, (2 * μ * w) = 2 * μ * ∑ w ∈ weights, w := by
    rw [← Finset.mul_sum]
  have h_expand : ∀ w ∈ weights, (w - μ) ^ 2 = w ^ 2 - 2 * μ * w + μ ^ 2 := by
    intro w _; ring
  have h_sum_expand : ∑ w ∈ weights, (w - μ) ^ 2 =
      S - 2 * μ * (∑ w ∈ weights, w) + (weights.card : ℝ) * μ ^ 2 := by
    calc ∑ w ∈ weights, (w - μ) ^ 2
        = ∑ w ∈ weights, (w ^ 2 - 2 * μ * w + μ ^ 2) := by
          apply Finset.sum_congr rfl; intro w hw; exact h_expand w hw
      _ = (∑ w ∈ weights, (w ^ 2 - 2 * μ * w)) + ∑ w ∈ weights, μ ^ 2 := by
          rw [Finset.sum_add_distrib]
      _ = ((∑ w ∈ weights, w ^ 2) - ∑ w ∈ weights, (2 * μ * w)) + ∑ w ∈ weights, μ ^ 2 := by
          rw [Finset.sum_sub_distrib]
      _ = (S - (2 * μ * ∑ w ∈ weights, w)) + (weights.card : ℝ) * μ ^ 2 := by
          rw [h_sum_2mu_w, h_sum_mu_sq]
      _ = S - 2 * μ * (∑ w ∈ weights, w) + (weights.card : ℝ) * μ ^ 2 := by ring
  have h_sum_w_eq : ∑ w ∈ weights, w = 1 := h_sum_one
  have h_mu_sq : (weights.card : ℝ) * μ ^ 2 = 1 / (weights.card : ℝ) := by
    unfold μ; field_simp
  have h_S_eq : ∑ w ∈ weights, (w - μ) ^ 2 = S - 1 / (weights.card : ℝ) := by
    rw [h_sum_expand, h_sum_w_eq, h_mu_sq]
    unfold μ
    field_simp
    ring
  have h_sum_sq_nonneg : 0 ≤ ∑ w ∈ weights, (w - μ) ^ 2 := by
    apply Finset.sum_nonneg; intro w _; exact sq_nonneg _
  have hS_ge_inv_card : 1 / (weights.card : ℝ) ≤ S := by linarith
  have h_one_le_div : 1 ≤ 1 / S := by
    rw [one_le_div hS_pos]
    exact hS_le_one
  have h_div_le_card : 1 / S ≤ (weights.card : ℝ) := by
    rw [div_le_iff₀ hS_pos]
    have h_mul : 1 ≤ S * (weights.card : ℝ) := by
      have h1 : 1 / (weights.card : ℝ) * (weights.card : ℝ) = 1 := by field_simp
      nlinarith
    linarith
  exact ⟨h_one_le_div, h_div_le_card⟩
/-- ESS-gated refresh (ASMC Eq.7 Kong et al. 1994 `N_eff=(∑w)²/∑w²`, AR triggers at `N_eff<νN`; ancestry operational `ν=0.5` standard adaptive vs `0.95` aggressive): `ESS<N` means weights are skewed — some `w>1/card`. Finite core behind PBRF `C`-factor rejuvenation (reweight child views / MCMC move) without touching the immutable parent. -/
theorem ESS_low_implies_skewed (weights : Finset ℝ) (h_sum_one : ∑ w ∈ weights, w = 1) (h_nonneg : ∀ w ∈ weights, 0 ≤ w) (hne : weights.Nonempty) (hESS : ESS weights h_sum_one h_nonneg < (weights.card : ℝ)) :
    ∃ w ∈ weights, 1 / (weights.card : ℝ) < w := by
  have hcard_pos_nat : 0 < weights.card := Finset.card_pos.mpr hne
  have hcard_pos : (0 : ℝ) < (weights.card : ℝ) := Nat.cast_pos.mpr hcard_pos_nat
  have hcard_ne : (weights.card : ℝ) ≠ 0 := ne_of_gt hcard_pos
  set μ : ℝ := 1 / (weights.card : ℝ) with hμ
  have hμ_nonneg : 0 ≤ μ := by positivity
  -- S > 0 from sum_one (else all zero contradicts sum_one)
  have hS_pos : 0 < ∑ w ∈ weights, w ^ 2 := by
    have h_exists_pos : ∃ w ∈ weights, 0 < w := by
      by_contra h
      push_neg at h
      have h_all_zero : ∀ w ∈ weights, w = 0 := by
        intro w hw
        have h_le := h w hw
        have h_nn := h_nonneg w hw
        linarith
      have h_sum_zero : ∑ w ∈ weights, w = 0 :=
        Finset.sum_eq_zero (fun w hw => h_all_zero w hw)
      linarith
    obtain ⟨w0, hw0_mem, hw0_pos⟩ := h_exists_pos
    exact Finset.sum_pos' (fun w _ => sq_nonneg w) ⟨w0, hw0_mem, sq_pos_of_pos hw0_pos⟩
  by_contra hcon
  push_neg at hcon
  -- every weight ≤ μ, so w^2 ≤ μ*w and S ≤ μ*1 = μ
  have h_sq : ∀ w ∈ weights, w ^ 2 ≤ μ * w := by
    intro w hw
    have hnn := h_nonneg w hw
    have hle : w ≤ μ := hcon w hw
    calc w ^ 2 = w * w := by ring
      _ ≤ μ * w := by exact mul_le_mul_of_nonneg_right hle hnn
  have hS_le : ∑ w ∈ weights, w ^ 2 ≤ μ := by
    calc ∑ w ∈ weights, w ^ 2 ≤ ∑ w ∈ weights, μ * w := Finset.sum_le_sum (fun w hw => h_sq w hw)
      _ = μ * ∑ w ∈ weights, w := by rw [Finset.mul_sum]
      _ = μ := by rw [h_sum_one, mul_one]
  have hESS_ge : (weights.card : ℝ) ≤ ESS weights h_sum_one h_nonneg := by
    unfold ESS
    have hμ_eq : μ = 1 / (weights.card : ℝ) := rfl
    have h1 : 1 / (1 / (weights.card : ℝ)) ≤ 1 / (∑ w ∈ weights, w ^ 2) :=
      one_div_le_one_div_of_le hS_pos (by rw [← hμ_eq]; exact hS_le)
    have h2 : (1 : ℝ) / (1 / (weights.card : ℝ)) = (weights.card : ℝ) := by field_simp
    rw [h2] at h1
    exact h1
  linarith
/-- Sharper skew (Elvira et al. 2019 `ESShat = N/(1+CV²)` Eq.27: low ESS ⟺ high weight-CV; EssSmc scout. Generalizes `ESS_low_implies_skewed` from threshold `card` to any positive `m`: `ESS ≤ m` forces some weight `≥ 1/m` (contrapositive: all weights `< 1/m` give `∑w² < 1/m`, i.e. `ESS > m`, via `Finset.sum_lt_sum` with strictness from the positive-mass particle that `∑w = 1` guarantees). Corroborated by Scipedia collapse case (`ESS` close to 1) and Elvira §4.2 `1/max[w]` metric. -/
theorem ESS_le_implies_max_weight (weights : Finset ℝ) (h_sum_one : ∑ w ∈ weights, w = 1)
    (h_nonneg : ∀ w ∈ weights, 0 ≤ w) (_hne : weights.Nonempty)
    (m : ℝ) (hm : 0 < m) (hESS : ESS weights h_sum_one h_nonneg ≤ m) :
    ∃ w ∈ weights, 1 / m ≤ w := by
  have hm_ne : m ≠ 0 := ne_of_gt hm
  have hS_pos : 0 < ∑ w ∈ weights, w ^ 2 := by
    have h_exists_pos : ∃ w ∈ weights, 0 < w := by
      by_contra h
      push Not at h
      have h_all_zero : ∀ w ∈ weights, w = 0 := by
        intro w hw
        have h_le := h w hw
        have h_nn := h_nonneg w hw
        linarith
      have h_sum_zero : ∑ w ∈ weights, w = 0 :=
        Finset.sum_eq_zero (fun w hw => h_all_zero w hw)
      linarith
    obtain ⟨w0, hw0_mem, hw0_pos⟩ := h_exists_pos
    exact Finset.sum_pos' (fun w _ => sq_nonneg w) ⟨w0, hw0_mem, sq_pos_of_pos hw0_pos⟩
  by_contra hcon
  push Not at hcon
  -- all weights `< 1/m`: squares bounded above by `(1/m)*w`, strictly at `w0`
  have h_le : ∀ w ∈ weights, w ^ 2 ≤ (1 / m) * w := by
    intro w hw
    have hnn := h_nonneg w hw
    have hlt : w < 1 / m := hcon w hw
    calc w ^ 2 = w * w := by ring
      _ ≤ (1 / m) * w := by exact mul_le_mul_of_nonneg_right (le_of_lt hlt) hnn
  have h_lt_at : ∃ w ∈ weights, w ^ 2 < (1 / m) * w := by
    obtain ⟨w0, hw0_mem, hw0_pos⟩ : ∃ w ∈ weights, 0 < w := by
      by_contra h
      push Not at h
      have h_all_zero : ∀ w ∈ weights, w = 0 := by
        intro w hw
        have h_le := h w hw
        have h_nn := h_nonneg w hw
        linarith
      have h_sum_zero : ∑ w ∈ weights, w = 0 :=
        Finset.sum_eq_zero (fun w hw => h_all_zero w hw)
      linarith
    refine ⟨w0, hw0_mem, ?_⟩
    have hlt : w0 < 1 / m := hcon w0 hw0_mem
    calc w0 ^ 2 = w0 * w0 := by ring
      _ < (1 / m) * w0 := by exact mul_lt_mul_of_pos_right hlt hw0_pos
  have hS_lt : ∑ w ∈ weights, w ^ 2 < 1 / m := by
    have hsum : ∑ w ∈ weights, (1 / m) * w = (1 / m) * ∑ w ∈ weights, w := by
      rw [Finset.mul_sum]
    calc ∑ w ∈ weights, w ^ 2 < ∑ w ∈ weights, (1 / m) * w :=
          Finset.sum_lt_sum h_le h_lt_at
      _ = (1 / m) * ∑ w ∈ weights, w := hsum
      _ = 1 / m := by rw [h_sum_one, mul_one]
  have hESS_gt : m < ESS weights h_sum_one h_nonneg := by
    unfold ESS
    have h1 : 1 / (1 / m) < 1 / (∑ w ∈ weights, w ^ 2) :=
      one_div_lt_one_div_of_lt hS_pos hS_lt
    have h2 : (1 : ℝ) / (1 / m) = m := by field_simp
    rw [h2] at h1
    exact h1
  linarith
structure Artifact where
  targetId : Nat
  provenanceEpoch : Nat

def artifactValid {Packet : Type} [DecidableEq Packet] (a : Artifact) (currentEpoch : Nat) (realPacket : Packet) (childPacket : Packet) : Prop :=
  a.provenanceEpoch = currentEpoch ∧ realPacket = childPacket

theorem stale_artifact_rejected
    {Packet : Type} [DecidableEq Packet]
    (a : Artifact) (currentEpoch : Nat) (realPacket childPacket : Packet)
    (hStale : a.provenanceEpoch ≠ currentEpoch)
    : ¬ artifactValid (Packet:=Packet) a currentEpoch realPacket childPacket := by
  unfold artifactValid; intro ⟨hEq, _⟩; exact hStale hEq

def siblingSquashed {Packet : Type} [DecidableEq Packet] (queriedPacket realPacket : Packet) : Prop := queriedPacket ≠ realPacket

end PBRFCore

end Hydra2.Blueprint.PBRF
