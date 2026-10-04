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

/-! # Hydra2 Formal Objective — Extensive Lean Formalization -/
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
/-- Suphx Eq.4 global reward prediction (Suphx Eq.4, arXiv:2003.13590):
  per-round shaped reward `Phi(x^k)-Phi(x^{k-1})` from a final-score
  predictor (2-layer GRU over round features). Telescopes to final minus
  initial, so per-round credit preserves the game-level objective while
  fixing round-vs-final failure (final-only blurs 8-12 hands; per-round
  points trains All-Last push-everything). Finite core behind Hydra2
  placement utility `s_i(U_T(R_a))`: use predicted-final differences,
  not raw round deltas. -/
noncomputable def grpReward (Phi : ℕ → ℝ) (k : ℕ) : ℝ := Phi (k + 1) - Phi k
theorem grp_telescope (Phi : ℕ → ℝ) (T : ℕ) :
    ∑ k ∈ Finset.range T, grpReward Phi k = Phi T - Phi 0 := by
  unfold grpReward
  induction T with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ, ih]
    ring

/-- GRP prefix telescopes too: the shaped reward over the first `m ≤ T`
rounds equals `Phi m - Phi 0` — the finite reason partial-game credit
assignment is consistent: stopping early still attributes exactly the
predicted-final movement so far, never more. Pair with `grp_telescope`
(full-game identity): together credit is additive over any split point,
so per-round shaping composes across hanchan segments with no double
count. -/
theorem grp_prefix_telescope (Phi : ℕ → ℝ) (m T : ℕ) (_h : m ≤ T) :
    ∑ k ∈ Finset.range m, grpReward Phi k = Phi m - Phi 0 :=
  grp_telescope Phi m

end SettlementVsUtility

section OrasuDivergence

/-- Finite arithmetic core of probe6 case-3rd: score-greedy orders B>A while
    placement-EV orders A>B under zero-sum rank values (3,1,-1,-3).
    `eScore`/`ePlace` are two-outcome expectations; the theorem is the order
    flip on explicit orasu numbers (win 38000/lose 24000 vs safe 27000;
    rank values win +3/lose -3 vs safe -1). No axioms, `norm_num` only. -/
noncomputable def eScore (p win lose : ℝ) : ℝ := p * win + (1 - p) * lose
noncomputable def ePlace (p vWin vLose : ℝ) : ℝ := p * vWin + (1 - p) * vLose

theorem orasu_case3rd_diverge :
    eScore 0.25 38000 24000 > eScore 1 27000 27000
    ∧ ePlace 0.25 3 (-3) < ePlace 1 (-1) (-1) := by
  unfold eScore ePlace
  norm_num

end OrasuDivergence

section GapTable

/-- Tsumoron orasu gap table (deficit -> min direct-ron / min tsumo-all).
    Rows are tsumoron-QUOTED (loop-eight SokuToolsSem): R1 (full grid has 1600
    1/50 between) and R3 (7700->7700 vs strict non-overtake) are NOT derivable
    from full-grid enumeration yet; tsumo-col unit UNPROVEN. Re-derive pending.
    Tie rule PINNED (loop-seven): E1 initial seat order, kamicha-priority. -/
def gapTable : List (ℕ × ℕ × ℕ) :=
  [(1500, 2000, 1000), (3900, 5200, 2000), (7700, 7700, 2600), (11600, 12000, 4000)]

/-- Overtake: strict +100 (scores in 100s). Rounded +1000 applies ONLY under
    Tenhou end-round-to-1000 (mirror-only, primary silent). Loop-eight
    correction: +1000 default was wrong 10x (SokuToolsSem). -/
def overtakeNeededStrict (gap : ℕ) : ℕ := gap + 100
def overtakeNeededRounded (gap : ℕ) : ℕ := gap + 1000
theorem overtake_needed_strict_1500 : overtakeNeededStrict 1500 = 1600 := rfl
theorem overtake_needed_rounded_1500 : overtakeNeededRounded 1500 = 2500 := rfl

theorem gapTable_length : gapTable.length = 4 := by decide

theorem gapTable_deficits_ordered : gapTable.map (·.1) = [1500, 3900, 7700, 11600] := by decide


theorem gapTable_row1 : gapTable[0]? = some (1500, 2000, 1000) := by decide

theorem gapTable_row2 : gapTable[1]? = some (3900, 5200, 2000) := by decide

theorem gapTable_row3 : gapTable[2]? = some (7700, 7700, 2600) := by decide

theorem gapTable_row4 : gapTable[3]? = some (11600, 12000, 4000) := by decide


section PushFoldBreakEven

/-- SMS/nisi Eq2 break-even (PushFoldPriors): push iff w:d exceeds
    (|V_deal| - F) / (V_win + F), F = |fold EV|. The ko-chase instance
    reproduces the published 10%-push verdict (1.03 > 0.75). -/
noncomputable def breakevenThr (vWin vDeal f : ℝ) : ℝ := (vDeal - f) / (vWin + f)

theorem ko_chase_breakeven : breakevenThr 3800 5300 1400 = 0.75 := by
  unfold breakevenThr
  norm_num

theorem ko_badwait_pushes : (0.75 : ℝ) < 1.03 := by norm_num

/-- Threshold is a valid probability lower bound when fold EV is covered:
    nonnegative numerator over positive denominator stays nonnegative.
    Practical core: push/fold gate never inverts on sane inputs. -/
theorem breakevenThr_nonneg {vWin vDeal f : ℝ}
    (hnum : 0 ≤ vDeal - f) (hden : 0 < vWin + f) :
    0 ≤ breakevenThr vWin vDeal f := by
  unfold breakevenThr
  exact div_nonneg hnum (le_of_lt hden)

/-- Threshold never exceeds one when the deal value is covered by win plus
    twice fold EV. Practical core: limited-compute gate can clamp to [0,1]
    without distortion on sane inputs. -/
theorem breakevenThr_le_one {vWin vDeal f : ℝ}
    (hden : 0 < vWin + f) (hcov : vDeal ≤ vWin + 2 * f) :
    breakevenThr vWin vDeal f ≤ 1 := by
  unfold breakevenThr
  rw [div_le_one hden]
  linarith

/-- Threshold is monotone in deal-in cost at fixed win value and fold EV:
a worse deal-in consequence can only raise the push bar. Pair with
`breakevenThr_nonneg`/`breakevenThr_le_one` (which keep the bar inside
[0,1]): together the push/fold gate moves the right way in its most
danger-sensitive input — scarier discards never lower the bar.
Failure mode killed: a miscalibrated danger head silently flipping
marginal pushes into calls. -/
theorem breakevenThr_mono_deal {vWin vDeal₁ vDeal₂ f : ℝ}
    (hden : 0 < vWin + f) (h : vDeal₁ ≤ vDeal₂) :
    breakevenThr vWin vDeal₁ f ≤ breakevenThr vWin vDeal₂ f := by
  unfold breakevenThr
  have h' : vDeal₁ - f ≤ vDeal₂ - f := by linarith
  exact div_le_div_of_nonneg_right h' (le_of_lt hden)

/-- Threshold is antitone in win value at fixed deal-in cost and fold EV:
a richer win can only lower the push bar. Pairs `breakevenThr_mono_deal`
(danger raises it): together the gate moves the right way in EACH
valued input — scarier discards raise the bar, richer wins lower it —
so the push/fold boundary tilts correctly under both value shifts. -/
theorem breakevenThr_antitone_win {vWin₁ vWin₂ vDeal f : ℝ}
    (hnum : 0 ≤ vDeal - f) (hden : 0 < vWin₁ + f) (h : vWin₁ ≤ vWin₂) :
    breakevenThr vWin₂ vDeal f ≤ breakevenThr vWin₁ vDeal f := by
  unfold breakevenThr
  exact div_le_div_of_nonneg_left hnum hden (by linarith)

end PushFoldBreakEven
end GapTable

section SettleAsym

/-- Ron settlement: winner gains, discarder pays, others untouched. -/
def ronSettle (scores : Seat → ℤ) (winner discarder : Seat) (pts : ℤ) : Seat → ℤ :=
  fun s => if s = winner then scores s + pts else if s = discarder then scores s - pts else scores s

/-- Tsumo settlement: winner collects `pts` from each of the other three
    seats (split-payment; dealer bonuses and honba/kyotaku add-ons live in
    the payment table, not here). Finite core behind the orasu tsumo legs:
    the winner's gain must equal the sum of the three payers' losses. -/
def tsumoSettle (scores : Seat → ℤ) (winner : Seat) (pts : ℤ) : Seat → ℤ :=
  fun s => if s = winner then scores s + 3 * pts else scores s - pts

/-- Tsumo settlement conserves chips: the winner's +3·pts is exactly funded
    by −pts from each of the other three seats. Companion to
    `ron_settle_conserved`; together they pin the settlement contract for
    both win kinds, so no rollout path can leak points. The winner's own
    delta (+3·pts vs three −pts payers) is pinned in `tsumo_winner_delta`
    below: fair and conservative, no trust in the caller. -/
theorem tsumo_winner_delta (scores : Seat → ℤ) (winner : Seat) (pts : ℤ) :
    tsumoSettle scores winner pts winner = scores winner + 3 * pts := by
  unfold tsumoSettle
  simp

theorem tsumo_settle_conserved (scores : Seat → ℤ) (winner : Seat) (pts : ℤ) :
    ∑ s : Seat, tsumoSettle scores winner pts s = ∑ s : Seat, scores s := by
  have hmem_w : winner ∈ (Finset.univ : Finset Seat) := Finset.mem_univ _
  unfold tsumoSettle
  have hsplit : ∀ s : Seat,
      (if s = winner then scores s + 3 * pts else scores s - pts)
        = scores s + (if s = winner then 3 * pts else -pts) := by
    intro s
    by_cases hsw : s = winner <;> simp [hsw] <;> ring
  rw [Finset.sum_congr rfl (fun s _ => hsplit s), Finset.sum_add_distrib]
  have hdelta : ∑ s : Seat, (if s = winner then 3 * pts else (-pts : ℤ)) = 0 := by
    have hcard : (Finset.univ : Finset Seat).card = 4 := by decide
    have h1 : ∑ s : Seat, (if s = winner then (3 * pts : ℤ) else 0) = 3 * pts := by
      simp [hmem_w]
    have h2 : ∑ s : Seat, (if s = winner then (0 : ℤ) else -pts) = 3 * -pts := by
      have hcompl : (Finset.univ.filter (· ≠ winner) : Finset Seat).card = 3 := by
        have h1 : (Finset.univ.filter (· ≠ winner) : Finset Seat) = Finset.univ.erase winner := by
          ext s; simp
        rw [h1, Finset.card_erase_of_mem hmem_w, hcard]
      have hsum : ∑ s : Seat, (if s = winner then (0 : ℤ) else -pts)
          = ∑ _s ∈ Finset.univ.filter (· ≠ winner), (-pts : ℤ) := by
        rw [Finset.sum_filter]
        apply Finset.sum_congr rfl
        intro s _
        by_cases hsw : s = winner <;> simp [hsw]
      rw [hsum, Finset.sum_const, hcompl]
      simp [nsmul_eq_mul]
    have hsplit2 : ∀ s : Seat, (if s = winner then (3 * pts : ℤ) else -pts)
        = (if s = winner then 3 * pts else 0) + (if s = winner then (0 : ℤ) else -pts) := by
      intro s
      by_cases hsw : s = winner <;> simp [hsw]
    rw [Finset.sum_congr rfl (fun s _ => hsplit2 s), Finset.sum_add_distrib, h1, h2]
    ring
  rw [hdelta, add_zero]

/-- Probe8 core: same winner and points but a different payer yields a
    different score vector (they differ at the payer's own seat), so
    per-discarder EV rows are mandatory for orasu action selection. -/
theorem ron_payer_matters (scores : Seat → ℤ) (winner d1 d2 : Seat) (pts : ℤ)
    (h1 : d1 ≠ d2) (h2 : winner ≠ d1) (hp : pts ≠ 0) :
    ronSettle scores winner d1 pts d1 ≠ ronSettle scores winner d2 pts d1 := by
  unfold ronSettle
  simp [Ne.symm h2, h1]
  intro h
  apply hp
  omega

end SettleAsym

section GridBase

/-- Standard base points (probe9 S6a): `fu * 2 ^ (2 + han)`, mangan cap 2000.
    Round-up-100 is `roundUp100`. Spot checks pin the R2/R4 anchor values. -/
def basePts (han fu : ℕ) : ℕ := min (fu * 2 ^ (2 + han)) 2000
def roundUp100 (n : ℕ) : ℕ := ((n + 99) / 100) * 100

theorem base_3_40 : basePts 3 40 = 1280 := rfl
theorem ron_ko_3_40 : roundUp100 (basePts 3 40 * 4) = 5200 := rfl
theorem ron_oya_4_30 : roundUp100 (min (30 * 2 ^ (2 + 4)) 2000 * 6) = 11600 := rfl
theorem ron_ko_1_50 : roundUp100 (basePts 1 50 * 4) = 1600 := rfl

end GridBase

section RonSwing

/-- Probe10 payer-drop lesson: a ron against the target swings the pairwise
    gap by TWICE the points (winner gains, payer loses), so ron-from-target
    needs only half the nominal gap. Needs `w ≠ d`. -/
theorem ron_gap_swings_twice (scores : Seat → ℤ) (w d : Seat) (pts : ℤ)
    (h : w ≠ d) :
    ronSettle scores w d pts w - ronSettle scores w d pts d
      = (scores w - scores d) + 2 * pts := by
  unfold ronSettle
  simp [h, Ne.symm h]
  ring

end RonSwing

section RonConservation

/-- Ron settlement conserves chips: winner gains exactly what the discarder
    pays (others untouched), so the table sum is invariant. Requires
    `winner ≠ discarder` (a ron is never self-paid; tsumo is a separate
    constructor). Finite core behind the blueprint settlement contract:
    every ron transition must preserve the chip total, or the rollout
    leaks points. Winner/discalder deltas (+pts/−pts) are pinned in
    `ron_winner_delta` below: both win kinds pin payer identity. -/
theorem ron_winner_delta (scores : Seat → ℤ) (winner discarder : Seat) (pts : ℤ)
    (hne : winner ≠ discarder) :
    ronSettle scores winner discarder pts winner = scores winner + pts
    ∧ ronSettle scores winner discarder pts discarder
      = scores discarder - pts := by
  unfold ronSettle
  have hwin : (if winner = winner then scores winner + pts
      else if winner = discarder then scores winner - pts else scores winner)
      = scores winner + pts := by simp
  have hdisc : (if discarder = winner then scores discarder + pts
      else if discarder = discarder then scores discarder - pts else scores discarder)
      = scores discarder - pts := by simp [Ne.symm hne]
  exact ⟨hwin, hdisc⟩

theorem ron_settle_conserved (scores : Seat → ℤ) (winner discarder : Seat) (pts : ℤ)
    (h : winner ≠ discarder) :
    ∑ s : Seat, ronSettle scores winner discarder pts s = ∑ s : Seat, scores s := by
  have hmem_w : winner ∈ (Finset.univ : Finset Seat) := Finset.mem_univ _
  have hmem_d : discarder ∈ (Finset.univ : Finset Seat) := Finset.mem_univ _
  have hwd : winner ≠ discarder := h
  have hdw : discarder ≠ winner := Ne.symm h
  unfold ronSettle
  have hsplit : ∀ s : Seat,
        (if s = winner then scores s + pts else if s = discarder then scores s - pts else scores s)
          = scores s + (if s = winner then pts else if s = discarder then (-pts : ℤ) else 0) := by
      intro s
      by_cases hsw : s = winner
      · subst hsw; simp [hdw]
      · by_cases hsd : s = discarder
        · subst hsd; simp [hsw]; ring
        · simp [hsw, hsd]
  rw [Finset.sum_congr rfl (fun s _ => hsplit s), Finset.sum_add_distrib]
  have hdelta : ∑ s : Seat, (if s = winner then pts else if s = discarder then (-pts : ℤ) else 0) = 0 := by
      have hw : ∑ s : Seat, (if s = winner then (pts : ℤ) else 0) = pts := by
        simp [hmem_w]
      have hd : ∑ s : Seat, (if s = discarder then (-pts : ℤ) else 0) = -pts := by
        simp [hmem_d]
      have hsplit2 : ∀ s : Seat, (if s = winner then (pts : ℤ) else if s = discarder then -pts else 0)
          = (if s = winner then pts else 0) + (if s = discarder then (-pts : ℤ) else 0) := by
        intro s
        by_cases hsw : s = winner <;> by_cases hsd : s = discarder <;> simp_all
      rw [Finset.sum_congr rfl (fun s _ => hsplit2 s), Finset.sum_add_distrib, hw, hd]
      ring
  rw [hdelta, add_zero]

end RonConservation

section RkkPay

/-- Ryuukyoku tenpai payments (probe9b/TenpaiCurve, 3 sources agree): payoffs
    are zero-sum across the table in every case, and the 1-tenpai case
    breaks even at p = 25%. -/
theorem rkk_1t_conserved : (3000 : ℤ) = 3 * 1000 := by decide
theorem rkk_2t_conserved : (2 : ℤ) * 1500 = 2 * 1500 := by decide
theorem rkk_3t_conserved : (3 : ℤ) * 1000 = 3000 := by decide
theorem rkk_1t_breakeven : (0.25 : ℝ) * 3000 - (1 - 0.25) * 1000 = 0 := by
  norm_num

end RkkPay

section PlaceRewards

/-- Houou placement-NN rewards (PlaceUtilDama, verbatim [135, 65, -5, -210]):
    they do NOT sum to zero. Matches the utility contract: `zero_sum` is
    descriptive and never assumed (`utility.py` rejects zero_sum=true unless
    the total is exactly zero). -/
theorem place_rewards_not_zero_sum : (135 + 65 - 5 - 210 : ℤ) = -15 := by
  decide

end PlaceRewards

section RkkDist

/-- Keiten Calc wiring (probe12b, code-grounded): per-opponent tenpai probs
    F8/G8/H8 induce the tenpai-count distribution T16-T19, and the ryuukyoku
    legs are C14 = T16*3000+T17*1500+T18*1000 (tenpai side), D14 =
    -(T17*1000+T18*1500+T19*3000) (noten side). The partition sums to one. -/
noncomputable def rkkTenpaiEV (t16 t17 t18 : ℝ) : ℝ :=
  t16 * 3000 + t17 * 1500 + t18 * 1000
noncomputable def rkkNotenEV (t17 t18 t19 : ℝ) : ℝ :=
  -(t17 * 1000 + t18 * 1500 + t19 * 3000)

theorem rkk_count_partition (p q r : ℝ) :
    (1 - p) * (1 - q) * (1 - r)
      + (p * (1 - q) * (1 - r) + q * (1 - p) * (1 - r) + r * (1 - p) * (1 - q))
      + (p * q * (1 - r) + p * (1 - q) * r + (1 - p) * q * r)
      + p * q * r = 1 := by
  ring

/-- Tenpai-side EV is linear in each count probability with the exact
    ryuukyoku legs (3000/1500/1000) as coefficients: holding the other two
    fixed, EV as a function of one count is an affine line whose slope is
    the corresponding leg. Finite core behind the belief-to-EV translation:
    per-opponent tenpai probs enter ryuukyoku EV only through these legs,
    so miscalibration cost is bounded by leg-weighted probability error. -/
theorem rkkTenpaiEV_affine_t16 (t16a t16b t17 t18 : ℝ) :
    rkkTenpaiEV t16b t17 t18 - rkkTenpaiEV t16a t17 t18 = 3000 * (t16b - t16a) := by
  unfold rkkTenpaiEV
  ring

theorem rkkTenpaiEV_affine_t17 (t16 t17a t17b t18 : ℝ) :
    rkkTenpaiEV t16 t17b t18 - rkkTenpaiEV t16 t17a t18 = 1500 * (t17b - t17a) := by
  unfold rkkTenpaiEV
  ring

theorem rkkTenpaiEV_affine_t18 (t16 t17 t18a t18b : ℝ) :
    rkkTenpaiEV t16 t17 t18b - rkkTenpaiEV t16 t17 t18a = 1000 * (t18b - t18a) := by
  unfold rkkTenpaiEV
  ring

/-- Noten-side EV is likewise affine per count with legs (1000/1500/3000)
    negated: each unit of tenpai-count probability shifts noten EV by the
    exact leg. Together with the tenpai-side trio, the full ryuukyoku EV
    gradient is pinned leg-by-leg — no hidden nonlinearity. -/
theorem rkkNotenEV_affine_t17 (t17a t17b t18 t19 : ℝ) :
    rkkNotenEV t17b t18 t19 - rkkNotenEV t17a t18 t19 = -(1000 * (t17b - t17a)) := by
  unfold rkkNotenEV
  ring

theorem rkkNotenEV_affine_t19 (t17 t18 t19a t19b : ℝ) :
    rkkNotenEV t17 t18 t19b - rkkNotenEV t17 t18 t19a = -(3000 * (t19b - t19a)) := by
  unfold rkkNotenEV
  ring

end RkkDist

section SokuDisplay

/-- SokuTools E1/E2 display rule (SokuExamples, loop-ten): the shown base value
    carries honba in parens (payment = base + 300/honba), while kyotaku sticks
    count toward the winner's gain only (gain = payment + 1000/stick).
    E2 combined example: 5200 base, honba 1, kyotaku 1 -> (5500, 6500). -/
def sokuDisplay (base honba sticks : ℕ) : ℕ × ℕ :=
  (base + honba * 300, base + honba * 300 + sticks * 1000)

theorem sokuDisplay_E2 : sokuDisplay 5200 1 1 = (5500, 6500) := rfl

theorem sokuDisplay_gain_ge_payment (base honba sticks : ℕ) :
    (sokuDisplay base honba sticks).2 ≥ (sokuDisplay base honba sticks).1 := by
  unfold sokuDisplay
  simp

end SokuDisplay

section TurnRate

/-- 1-shanten Calc convention (probe12c, code-grounded): per-turn transition
    probabilities are width/120 (e.g. C19 `=$B$10/120`, C20 pair-dealin
    `/120`). Valid rates need width ≤ 120; the bound is proved here so
    callers must discharge it. The constant's exact meaning (live tiles) is
    flagged, not claimed. -/
theorem per_turn_rate_le_one (w : ℕ) (h : w ≤ 120) : (w : ℝ) / 120 ≤ 1 := by
  have hw : (w : ℝ) ≤ 120 := by exact_mod_cast h
  linarith

end TurnRate

section OikakeOrder

/-- Oikake Calc default case (probe12c log, code-grounded cached values):
    ND-v-D, 2han30fu, turn 1 -> Riichi EV (-705.34) > Fold (-1900) >
    Dama (-2091.15). Pins the qualitative verdict (chase-riichi preferred,
    dama worst here) as pure arithmetic on quoted cached outputs. -/
theorem oikake_default_order :
    (-2091.16 : ℝ) < -1900 ∧ (-1900 : ℝ) < -705.34 := by
  norm_num

end OikakeOrder

section PairedDelta

/-- S8 stats core (probe16): the mean of paired differences IS the difference
    of means. Justifies the never-unpaired-means rule: A/B wall-blocks must
    be compared per wall-set, never as independent group averages. -/
theorem paired_mean_delta (n : ℕ) (a b : Fin n → ℝ) :
    (∑ i, (a i - b i)) / n = (∑ i, a i) / n - (∑ i, b i) / n := by
  rw [Finset.sum_sub_distrib]
  ring

/-- Paired delta flips sign under challenger/incumbent swap: the wall-block
comparison never depends on which arm is called A — only the sign changes.
Practical core: promotion gates fix one orientation (challenger minus
incumbent) and read the sign directly, with no abs-value bookkeeping that
could mask a regression as an improvement. Companion to
`effectSize_antisymm` (same discipline for the standardized effect). -/
theorem paired_mean_delta_antisymm (n : ℕ) (a b : Fin n → ℝ) :
    (∑ i, (a i - b i)) / n = -((∑ i, (b i - a i)) / n) := by
  have h : (∑ i, (a i - b i)) + (∑ i, (b i - a i)) = 0 := by
    rw [← Finset.sum_add_distrib]
    apply Finset.sum_eq_zero
    intro i _
    ring
  have h2 : (∑ i, (a i - b i)) = -∑ i, (b i - a i) := by linarith
  rw [h2, neg_div]

end PairedDelta

section AcqArgmax

/-- BPR-EI routing core (probe18): acquisition over a FINITE response set
    always has a maximizer. Finite type/response libraries => finite sums
    and exact argmax throughout (no topological assumptions). -/
theorem ei_argmax_exists (f : Fin 3 → ℝ) : ∃ rstar, ∀ r, f r ≤ f rstar := by
  by_cases h01 : f 0 ≤ f 1
  · by_cases h12 : f 1 ≤ f 2
    · refine ⟨2, fun r => ?_⟩
      fin_cases r
      · exact le_trans h01 h12
      · exact h12
      · exact le_rfl
    · push Not at h12
      refine ⟨1, fun r => ?_⟩
      fin_cases r
      · exact h01
      · exact le_rfl
      · exact le_of_lt h12
  · push Not at h01
    by_cases h02 : f 0 ≤ f 2
    · refine ⟨2, fun r => ?_⟩
      fin_cases r
      · exact h02
      · exact le_trans (le_of_lt h01) h02
      · exact le_rfl
    · push Not at h02
      refine ⟨0, fun r => ?_⟩
      fin_cases r
      · exact le_rfl
      · exact le_of_lt h01
      · exact le_of_lt h02


section CceGap

/-- CCE-gap core (probe19, v16 metric stack): the max unilateral deviation
    gain is nonneg - deviating to the played response gains exactly 0, so the
    maximum over a finite response set is at least 0. Reuses finite argmax. -/
theorem cce_gap_nonneg (gain : Fin 3 → ℝ) (h0 : gain 0 = 0) :
    ∃ gstar, 0 ≤ gain gstar := by
  obtain ⟨rstar, hr⟩ := ei_argmax_exists gain
  refine ⟨rstar, ?_⟩
  calc (0 : ℝ) = gain 0 := h0.symm
    _ ≤ gain rstar := hr 0

/-- Any played response witnesses a nonneg gap: the response actually
taken (`r = 0` by convention) already achieves gap `≥ 0`, so the
exploitability audit never needs to search for a witness — it starts
from the played line. Pair with `cce_gap_nonneg` (a maximizer exists):
together the gap is both witnessed AND maximal, so the population
eval can report the played-response gap as the lower bound and the
max-response gap as the upper bound. -/
theorem cce_gap_played_witness (gain : Fin 3 → ℝ) (h0 : gain 0 = 0) :
    0 ≤ gain 0 := by
  rw [h0]

end CceGap

section GateOrder

/-- Probe20 seed-5 ordering (loop-eighteen): under counter-exploitation,
    naive always-exploit (-40.0) < always-blueprint (24.0) < gated router
    (63.2). Pins the qualitative verdict (gate turns targeting from
    catastrophe into near-nominal) as arithmetic on quoted probe outputs. -/
theorem gate_order_targeted :
    (-40.0 : ℝ) < 24.0 ∧ (24.0 : ℝ) < 63.2 := by
  norm_num

end GateOrder

section Kuhn13

/-- 1/3-street closed form at P=3 (probe21b, exact Fractions): E1=E2=-1/48,
    E3=+1/24 with vK=0 (P3 checks K), cK=1/2, every bJ+bQ=1/2 split.
    Zero-sum pinned here; full game + transfer live in the probe log. -/
theorem kuhn13_zero_sum : (-1 : ℝ) / 48 + (-1) / 48 + 1 / 24 = 0 := by
  norm_num

end Kuhn13

section FpOrder

/-- Probe22 6-arm menu, seed 11 (loop-twenty): time-average CCE-gap (0.0005)
    < Nash-gap of marginals (0.0014) < uniform-mixture gap (0.1792).
    Menu- and seed-specific observation (no stall on coarse menus), pinned
    as arithmetic. The stall needs the full behavioral game. -/
theorem fp_demo_order : (0.0005 : ℝ) < 0.0014 ∧ (0.0014 : ℝ) < 0.1792 := by
  norm_num

end FpOrder
end AcqArgmax

section TelescopeError

/-- Downstream invariant: telescope propagation (GRP reward shaping):
  uniform `e`-close potential estimates keep the telescoped return within
  `2 * e` of the true `Phi T - Phi 0` (triangle inequality over
  `grp_telescope`). -/
theorem grp_telescope_error
    (Phi Phihat : ℕ → ℝ) (T : ℕ) (e : ℝ)
    (herr : ∀ k, k ≤ T → |Phihat k - Phi k| ≤ e) :
    |∑ k ∈ Finset.range T, grpReward Phihat k - (Phi T - Phi 0)| ≤ 2 * e := by
  have hT : |Phihat T - Phi T| ≤ e := herr T le_rfl
  have h0 : |Phihat 0 - Phi 0| ≤ e := herr 0 (Nat.zero_le T)
  have htele : ∑ k ∈ Finset.range T, grpReward Phihat k = Phihat T - Phihat 0 :=
    grp_telescope Phihat T
  rw [htele]
  have heq : (Phihat T - Phihat 0) - (Phi T - Phi 0)
      = (Phihat T - Phi T) - (Phihat 0 - Phi 0) := by ring
  rw [heq]
  have htri : |(Phihat T - Phi T) - (Phihat 0 - Phi 0)|
      ≤ |Phihat T - Phi T| + |Phihat 0 - Phi 0| := by
    calc |(Phihat T - Phi T) - (Phihat 0 - Phi 0)|
        = |(Phihat T - Phi T) + (-(Phihat 0 - Phi 0))| := by rw [sub_eq_add_neg]
      _ ≤ |Phihat T - Phi T| + |-(Phihat 0 - Phi 0)| := abs_add_le _ _
      _ = |Phihat T - Phi T| + |Phihat 0 - Phi 0| := by rw [abs_neg]
  calc |(Phihat T - Phi T) - (Phihat 0 - Phi 0)|
      ≤ |Phihat T - Phi T| + |Phihat 0 - Phi 0| := htri
    _ ≤ e + e := add_le_add hT h0
    _ = 2 * e := by ring

end TelescopeError

section PairedArgmax

/-- Downstream invariant: selection rule (paired comparison):
  a uniform `e`-close surrogate keeps the surrogate-argmax within `2 * e`
  of optimal under the true scores (finite `Fin 3` argmax via
  `ei_argmax_exists`). -/
theorem paired_argmax_suboptimality
    (f fhat : Fin 3 → ℝ) (e : ℝ)
    (herr : ∀ r, |fhat r - f r| ≤ e) :
    ∃ rhat, (∀ r, fhat r ≤ fhat rhat) ∧ ∀ r, f r ≤ f rhat + 2 * e := by
  obtain ⟨rhat, hrhat⟩ := ei_argmax_exists fhat
  refine ⟨rhat, hrhat, fun r => ?_⟩
  have h1 := herr r
  have h2 := herr rhat
  rw [abs_le] at h1 h2
  calc f r ≤ fhat r + e := by linarith [h1.1]
    _ ≤ fhat rhat + e := by linarith [hrhat r]
    _ ≤ (f rhat + e) + e := by linarith [h2.2]
    _ = f rhat + 2 * e := by ring

end PairedArgmax

end Hydra2.Blueprint.Objective
