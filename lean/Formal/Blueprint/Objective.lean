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
set_option linter.style.whitespace false

/-! # Hydra2 §3 Formal Objective — Extensive Lean Formalization -/
namespace Hydra2.Blueprint.Objective

abbrev Seat := Fin 4

structure UtilityVector where
  vals : Seat → ℝ


structure RawOutcome where
  scores : Seat → ℤ
  ranks : Seat → Fin 4

def rootScalar (u : UtilityVector) (i : Seat) : ℝ := u.vals i

section QDelta

variable {World : Type} [Fintype World] [DecidableEq World]
variable {Outcome : Type} [Fintype Outcome] [DecidableEq Outcome]

structure Belief (World : Type) [Fintype World] where
  prob : World → ℝ
  nonneg : ∀ x, 0 ≤ prob x
  sum_one : ∑ x : World, prob x = 1

def OutcomeDist (World Outcome : Type) [Fintype World] [Fintype Outcome] :=
  World → Outcome → ℝ

noncomputable def Q_value
    (b : Belief World)
    (kernel : OutcomeDist World Outcome)
    (utility : Outcome → UtilityVector)
    (seat : Seat) : ℝ :=
  ∑ x : World, ∑ o : Outcome, b.prob x * kernel x o * rootScalar (utility o) seat

noncomputable def Delta
    (b : Belief World)
    (kernel_a kernel_b : OutcomeDist World Outcome)
    (utility : Outcome → UtilityVector)
    (seat : Seat) : ℝ :=
  Q_value b kernel_a utility seat - Q_value b kernel_b utility seat

structure Coupling (Outcome : Type) [Fintype Outcome] where
  joint : Outcome → Outcome → ℝ
  nonneg : ∀ a b, 0 ≤ joint a b
  sum_one : ∑ a : Outcome, ∑ b : Outcome, joint a b = 1

def isCorrectCoupling
    (b : Belief World)
    (kA kB : OutcomeDist World Outcome)
    (Γ : Coupling Outcome) : Prop :=
  (∀ oa : Outcome, ∑ ob : Outcome, Γ.joint oa ob = ∑ x : World, b.prob x * kA x oa)
  ∧ (∀ ob : Outcome, ∑ oa : Outcome, Γ.joint oa ob = ∑ x : World, b.prob x * kB x ob)

noncomputable def couplingDelta
    (Γ : Coupling Outcome)
    (utility : Outcome → UtilityVector)
    (seat : Seat) : ℝ :=
  ∑ oa : Outcome, ∑ ob : Outcome, Γ.joint oa ob *
    (rootScalar (utility oa) seat - rootScalar (utility ob) seat)

theorem coupling_preserves_delta
    (b : Belief World)
    (kA kB : OutcomeDist World Outcome)
    (utility : Outcome → UtilityVector)
    (seat : Seat)
    (Γ : Coupling Outcome)
    (hΓ : isCorrectCoupling b kA kB Γ) :
    couplingDelta Γ utility seat = Delta b kA kB utility seat := by
  unfold couplingDelta Delta
  have h1 := hΓ.1
  have h2 := hΓ.2
  have eq : couplingDelta Γ utility seat =
    (∑ oa : Outcome, (∑ ob : Outcome, Γ.joint oa ob) * rootScalar (utility oa) seat)
    - (∑ ob : Outcome, (∑ oa : Outcome, Γ.joint oa ob) * rootScalar (utility ob) seat) := by
    unfold couplingDelta
    simp_rw [mul_sub, Finset.sum_sub_distrib]
    congr 1
    · congr 1; ext oa
      rw [Finset.sum_mul]
    · rw [Finset.sum_comm]
      congr 1; ext ob
      rw [Finset.sum_mul]
  calc couplingDelta Γ utility seat
      = (∑ oa : Outcome, (∑ ob : Outcome, Γ.joint oa ob) * rootScalar (utility oa) seat)
        - (∑ ob : Outcome, (∑ oa : Outcome, Γ.joint oa ob) * rootScalar (utility ob) seat) := eq
    _ = (∑ oa : Outcome, (∑ x : World, b.prob x * kA x oa) * rootScalar (utility oa) seat)
        - (∑ ob : Outcome, (∑ x : World, b.prob x * kB x ob) * rootScalar (utility ob) seat) := by
          congr 1
          · congr 1; ext oa; rw [h1 oa]
          · congr 1; ext ob; rw [h2 ob]
    _ = Q_value b kA utility seat - Q_value b kB utility seat := by
          have hQA : (∑ oa : Outcome, (∑ x : World, b.prob x * kA x oa) * rootScalar (utility oa) seat)
              = Q_value b kA utility seat := by
            unfold Q_value
            rw [Finset.sum_comm]
            congr 1; ext oa
            rw [← Finset.sum_mul]
          have hQB : (∑ ob : Outcome, (∑ x : World, b.prob x * kB x ob) * rootScalar (utility ob) seat)
              = Q_value b kB utility seat := by
            unfold Q_value
            rw [Finset.sum_comm]
            congr 1; ext ob
            rw [← Finset.sum_mul]
          rw [hQA, hQB]
theorem Q_point_mass
    (w0 : World)
    (b : Belief World)
    (hb : ∀ x, b.prob x = if x = w0 then 1 else 0)
    (kernel : OutcomeDist World Outcome)
    (utility : Outcome → UtilityVector)
    (seat : Seat) :
    Q_value b kernel utility seat = ∑ o : Outcome, kernel w0 o * rootScalar (utility o) seat := by
  unfold Q_value
  have step : ∀ x o, b.prob x * kernel x o * rootScalar (utility o) seat =
              if x = w0 then kernel w0 o * rootScalar (utility o) seat else 0 := by
    intro x o; rw [hb x]; split
    · next h => subst h; ring
    · next h => simp [h]
  simp_rw [step]
  rw [Finset.sum_comm]
  have collapse : ∀ o : Outcome, ∑ x : World, (if x = w0 then kernel w0 o * rootScalar (utility o) seat else (0 : ℝ)) =
                  kernel w0 o * rootScalar (utility o) seat := by
    intro o
    have : ∑ x : World, (if x = w0 then kernel w0 o * rootScalar (utility o) seat else (0 : ℝ)) =
           ∑ x : World, (if x = w0 then 1 else 0 : ℝ) * (kernel w0 o * rootScalar (utility o) seat) := by
      congr 1; ext x; split <;> ring
    rw [this, ← Finset.sum_mul]
    have sum_one : ∑ x : World, (if x = w0 then (1:ℝ) else 0) = 1 := by
      have hmem : w0 ∈ (Finset.univ : Finset World) := Finset.mem_univ _
      rw [Finset.sum_ite_eq' Finset.univ w0 (fun _ => (1:ℝ))]
      simp [hmem]
    rw [sum_one, one_mul]
  simp_rw [collapse]
theorem clairvoyance_inequality_two_actions
    (Γ : Coupling Outcome)
    (va vb : Outcome → ℝ)
    (h_nonneg : ∀ a b, 0 ≤ Γ.joint a b) :
    ∑ a : Outcome, ∑ b : Outcome, Γ.joint a b * max (va a) (vb b)
    ≥ max (∑ a : Outcome, ∑ b : Outcome, Γ.joint a b * va a)
          (∑ a : Outcome, ∑ b : Outcome, Γ.joint a b * vb b) := by
  have h1 : ∑ a : Outcome, ∑ b : Outcome, Γ.joint a b * va a ≤
            ∑ a : Outcome, ∑ b : Outcome, Γ.joint a b * max (va a) (vb b) := by
    apply Finset.sum_le_sum; intro a _
    apply Finset.sum_le_sum; intro b _
    exact mul_le_mul_of_nonneg_left (le_max_left _ _) (h_nonneg a b)
  have h2 : ∑ a : Outcome, ∑ b : Outcome, Γ.joint a b * vb b ≤
            ∑ a : Outcome, ∑ b : Outcome, Γ.joint a b * max (va a) (vb b) := by
    apply Finset.sum_le_sum; intro a _
    apply Finset.sum_le_sum; intro b _
    exact mul_le_mul_of_nonneg_left (le_max_right _ _) (h_nonneg a b)
  exact max_le h1 h2

private lemma sum_bool_eq {α : Type} [AddCommMonoid α] (f : Bool → α) :
    ∑ x : Bool, f x = f true + f false := by
  have h : (Finset.univ : Finset Bool) = {true, false} := by decide
  rw [h, Finset.sum_pair (by decide : (true:Bool) ≠ false)]

theorem clairvoyance_strict_gap_exists :
    ∃ (World2 : Type) (_ : Fintype World2) (_ : DecidableEq World2)
      (b : Belief World2) (va vb : World2 → ℝ),
      (max (∑ x : World2, b.prob x * va x) (∑ x : World2, b.prob x * vb x) <
       ∑ x : World2, b.prob x * max (va x) (vb x)) := by
  let World2 := Bool
  let b : Belief World2 := {
    prob := fun _ => 0.5
    nonneg := fun _ => by norm_num
    sum_one := by rw [sum_bool_eq]; norm_num
  }
  let va : World2 → ℝ := fun w => if w then 1 else 0
  let vb : World2 → ℝ := fun w => if w then 0 else 1
  refine ⟨World2, inferInstance, inferInstance, b, va, vb, ?_⟩
  have hva : ∑ x : World2, b.prob x * va x = 0.5 := by
    rw [sum_bool_eq]; simp [b, va]
  have hvb : ∑ x : World2, b.prob x * vb x = 0.5 := by
    rw [sum_bool_eq]; simp [b, vb]
  have hEmax : ∑ x : World2, b.prob x * max (va x) (vb x) = 1 := by
    have hmax : ∀ x : World2, max (va x) (vb x) = 1 := by intro x; cases x <;> simp [va, vb]
    simp_rw [hmax]
    have hsum : ∑ x : World2, b.prob x * (1:ℝ) = ∑ x : World2, b.prob x := by simp
    rw [hsum, b.sum_one]
  rw [hva, hvb, max_self, hEmax]; norm_num

theorem coupling_marginal_necessity :
    ∃ (World0 : Type) (_ : Fintype World0) (_ : DecidableEq World0)
        (Outcome0 : Type) (_ : Fintype Outcome0) (_ : DecidableEq Outcome0)
        (b : Belief World0) (kA kB : OutcomeDist World0 Outcome0)
        (utility : Outcome0 → UtilityVector) (seat : Seat)
        (Γ : Coupling Outcome0),
        ¬ isCorrectCoupling (World:=World0) b kA kB Γ ∧
        couplingDelta (Outcome:=Outcome0) Γ utility seat ≠ Delta (World:=World0) b kA kB utility seat := by
  refine ⟨Unit, inferInstance, inferInstance, Bool, inferInstance, inferInstance, ?_⟩
  let b : Belief Unit := { prob := fun _ => 1, nonneg := fun _ => by norm_num, sum_one := by simp }
  let kA : OutcomeDist Unit Bool := fun _ o => if o = true then 1 else 0
  let kB : OutcomeDist Unit Bool := fun _ o => if o = false then 1 else 0
  let utility : Bool → UtilityVector := fun b => ⟨fun s => if s = 0 ∧ b = true then 1 else 0⟩
  let seat : Seat := 0
  let Γ : Coupling Bool := {
    joint := fun a b => if a == false && b == false then 1 else 0
    nonneg := by intros a b; cases a <;> cases b <;> simp <;> norm_num
    sum_one := by
      simp only [sum_bool_eq]
      simp
  }
  refine ⟨b, kA, kB, utility, seat, Γ, ?_, ?_⟩
  · intro h
    have h1 := h.1
    have htrue := h1 true
    have left : (∑ ob : Bool, Γ.joint true ob) = 0 := by simp [Γ, sum_bool_eq]
    have right : (∑ x : Unit, b.prob x * kA x true) = 1 := by simp [b, kA]
    rw [left, right] at htrue; norm_num at htrue
  · have hDelta : Delta (World:=Unit) b kA kB utility seat = 1 := by
      unfold Delta Q_value rootScalar
      simp [b, kA, kB, utility, seat, sum_bool_eq]
    have hCoup : couplingDelta (Outcome:=Bool) Γ utility seat = 0 := by
      unfold couplingDelta rootScalar
      simp [Γ, utility, seat, sum_bool_eq]
    rw [hDelta, hCoup]; norm_num
theorem Delta_linear_shared_randomness
    (b : Belief World)
    (kA kB : OutcomeDist World Outcome)
    (utility : Outcome → UtilityVector) (seat : Seat)
    (Γ1 Γ2 : Coupling Outcome)
    (h1 : isCorrectCoupling b kA kB Γ1)
    (h2 : isCorrectCoupling b kA kB Γ2) :
    couplingDelta Γ1 utility seat = couplingDelta Γ2 utility seat := by
  rw [coupling_preserves_delta b kA kB utility seat Γ1 h1,
      coupling_preserves_delta b kA kB utility seat Γ2 h2]

theorem utility_not_zero_sum_counterexample :
    ∃ (u : UtilityVector), ∑ i : Seat, u.vals i ≠ 0 := by
  use ⟨fun i => if i = 0 then 1 else 0⟩
  have h : ∑ i : Seat, (if i = (0:Seat) then (1:ℝ) else 0) = 1 := by
    have h0 : (0:Seat) ∈ (Finset.univ : Finset Seat) := Finset.mem_univ _
    rw [Finset.sum_ite_eq' Finset.univ (0:Seat) (fun _ => (1:ℝ))]
    simp [h0]
  simp [h]

end QDelta

section SettlementVsUtility
structure Settlement where
  deltas : Seat → ℤ
  conserved : ∑ i : Seat, deltas i = 0

theorem settlement_conserved_implies_not_utility_conserved :
    ∃ (u : UtilityVector) (s : Settlement), ∑ i : Seat, u.vals i ≠ 0 ∧ ∑ i : Seat, s.deltas i = 0 := by
  obtain ⟨u, hu⟩ := utility_not_zero_sum_counterexample
  exact ⟨u, ⟨fun _ => 0, by simp⟩, hu, by simp⟩

theorem settlement_zero_sum_always (s : Settlement) : (∑ i : Seat, (s.deltas i : ℝ)) = 0 := by
  have h := s.conserved
  exact_mod_cast h
/-- Suphx Eq.4 global reward prediction (SotaMahjong/SupxDetails/PpoRl scouts): per-round shaped reward `Phi(x^k)-Phi(x^{k-1})` from a final-score predictor (2-layer GRU over round features). Telescopes to final minus initial, so per-round credit preserves the game-level objective while fixing round-vs-final failure (final-only blurs 8-12 hands; per-round points trains All-Last push-everything). Finite core behind Hydra2 placement utility `s_i(U_T(R_a))`: use predicted-final differences, not raw round deltas. -/
noncomputable def grpReward (Phi : ℕ → ℝ) (k : ℕ) : ℝ := Phi (k + 1) - Phi k
theorem grp_telescope (Phi : ℕ → ℝ) (T : ℕ) :
    ∑ k ∈ Finset.range T, grpReward Phi k = Phi T - Phi 0 := by
  unfold grpReward
  induction T with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ, ih]
    ring

end SettlementVsUtility

end Hydra2.Blueprint.Objective
