import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import Formal.Blueprint.Objective

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedDecidableInType false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false

namespace Hydra2.Blueprint.Belief

section SuccessorBelief

variable {World : Type} [Fintype World] [DecidableEq World]
variable {Packet : Type} [Fintype Packet] [DecidableEq Packet]

structure TransKernel (World Packet : Type) [Fintype World] [Fintype Packet] where
  prob : World → World → Packet → ℝ
  nonneg : ∀ x x' e, 0 ≤ prob x x' e
  stochastic : ∀ x, ∑ x' : World, ∑ e : Packet, prob x x' e = 1

structure Belief (World : Type) [Fintype World] where
  prob : World → ℝ
  nonneg : ∀ x, 0 ≤ prob x
  sum_one : ∑ x : World, prob x = 1

noncomputable def jointP (b : Belief World) (K : TransKernel World Packet) : World → Packet → ℝ :=
  fun x' e => ∑ x : World, b.prob x * K.prob x x' e

noncomputable def Ze (b : Belief World) (K : TransKernel World Packet) (e : Packet) : ℝ :=
  ∑ x' : World, jointP b K x' e

noncomputable def packetMarginal (K : TransKernel World Packet) (x : World) (e : Packet) : ℝ :=
  ∑ x' : World, K.prob x x' e

theorem Ze_eq_weighted_marginal (b : Belief World) (K : TransKernel World Packet) (e : Packet) :
    Ze b K e = ∑ x : World, b.prob x * packetMarginal K x e := by
  unfold Ze jointP packetMarginal
  calc ∑ x' : World, ∑ x : World, b.prob x * K.prob x x' e
      = ∑ x : World, ∑ x' : World, b.prob x * K.prob x x' e := by rw [Finset.sum_comm]
    _ = ∑ x : World, b.prob x * ∑ x' : World, K.prob x x' e := by
        congr 1; ext x; rw [Finset.mul_sum]

theorem Ze_nonneg (b : Belief World) (K : TransKernel World Packet) (e : Packet) : 0 ≤ Ze b K e := by
  unfold Ze jointP
  apply Finset.sum_nonneg; intro x' _; apply Finset.sum_nonneg; intro x _; exact mul_nonneg (b.nonneg x) (K.nonneg x x' e)

theorem jointP_nonneg (b : Belief World) (K : TransKernel World Packet) (x' : World) (e : Packet) : 0 ≤ jointP b K x' e := by
  unfold jointP; apply Finset.sum_nonneg; intro x _; exact mul_nonneg (b.nonneg x) (K.nonneg x x' e)

theorem packet_partition_Z_sum_one (b : Belief World) (K : TransKernel World Packet) : ∑ e : Packet, Ze b K e = 1 := by
  unfold Ze jointP
  calc ∑ e : Packet, ∑ x' : World, ∑ x : World, b.prob x * K.prob x x' e
      = ∑ e : Packet, ∑ x : World, ∑ x' : World, b.prob x * K.prob x x' e := by congr 1; ext e; rw [Finset.sum_comm]
    _ = ∑ x : World, ∑ e : Packet, ∑ x' : World, b.prob x * K.prob x x' e := by rw [Finset.sum_comm]
    _ = ∑ x : World, ∑ e : Packet, b.prob x * ∑ x' : World, K.prob x x' e := by congr 1; ext x; congr 1; ext e; rw [Finset.mul_sum]
    _ = ∑ x : World, b.prob x * ∑ e : Packet, ∑ x' : World, K.prob x x' e := by congr 1; ext x; rw [Finset.mul_sum]
    _ = ∑ x : World, b.prob x * 1 := by
        congr 1; ext x; have hk := K.stochastic x
        have comm : ∑ e : Packet, ∑ x' : World, K.prob x x' e = ∑ x' : World, ∑ e : Packet, K.prob x x' e := by rw [Finset.sum_comm]
        rw [comm, hk]
    _ = ∑ x : World, b.prob x := by simp
    _ = 1 := b.sum_one

theorem jointP_total_one (b : Belief World) (K : TransKernel World Packet) : ∑ x' : World, ∑ e : Packet, jointP b K x' e = 1 := by
  have h := packet_partition_Z_sum_one b K; unfold Ze at h; rw [Finset.sum_comm] at h; exact h

noncomputable def successorBelief (b : Belief World) (K : TransKernel World Packet) (e : Packet) (he : Ze b K e ≠ 0) : World → ℝ :=
  fun x' => jointP b K x' e / Ze b K e

theorem successorBelief_nonneg (b : Belief World) (K : TransKernel World Packet) (e : Packet) (he : Ze b K e ≠ 0) : ∀ x', 0 ≤ successorBelief b K e he x' := by
  intro x'; unfold successorBelief; apply div_nonneg (jointP_nonneg b K x' e); have hpos : 0 ≤ Ze b K e := Ze_nonneg b K e; linarith

theorem successorBelief_sum_one (b : Belief World) (K : TransKernel World Packet) (e : Packet) (he : Ze b K e ≠ 0) : ∑ x' : World, successorBelief b K e he x' = 1 := by
  unfold successorBelief
  rw [← Finset.sum_div]
  exact div_self he

theorem impossible_packet_joint_zero (b : Belief World) (K : TransKernel World Packet) (e : Packet) (h0 : Ze b K e = 0) : ∀ x', jointP b K x' e = 0 := by
  intro x'
  have hsum : ∑ x' : World, jointP b K x' e = 0 := h0
  have hnn : ∀ x', 0 ≤ jointP b K x' e := fun x' => jointP_nonneg b K x' e
  have hle : jointP b K x' e ≤ ∑ x'' : World, jointP b K x'' e :=
    Finset.single_le_sum (fun y _ => hnn y) (Finset.mem_univ x')
  have hge : (0 : ℝ) ≤ jointP b K x' e := hnn x'
  have hle0 : jointP b K x' e ≤ 0 := by linarith
  exact le_antisymm hle0 hge

/-- Posterior packaged as a lawful `Belief`, so updates iterate without leaving
the structure. -/
noncomputable def successorBeliefPackaged (b : Belief World) (K : TransKernel World Packet)
    (e : Packet) (he : Ze b K e ≠ 0) : Belief World :=
  ⟨successorBelief b K e he, successorBelief_nonneg b K e he,
    successorBelief_sum_one b K e he⟩

/-- Tower / law of total expectation: reweighting the packaged posterior by `Ze`
recovers the joint (`Ze = 0` packets carry no mass by `impossible_packet_joint_zero`,
so the clean form quantifies over positive-`Ze` evidence). -/
theorem successor_tower (b : Belief World) (K : TransKernel World Packet) (g : World → ℝ)
    (hpos : ∀ e, Ze b K e ≠ 0) :
    ∑ e : Packet, Ze b K e * ∑ x' : World, successorBelief b K e (hpos e) x' * g x'
    = ∑ x' : World, ∑ e : Packet, jointP b K x' e * g x' := by
  have per : ∀ e : Packet, Ze b K e * ∑ x' : World, successorBelief b K e (hpos e) x' * g x'
      = ∑ x' : World, jointP b K x' e * g x' := by
    intro e
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun x' _ => ?_
    unfold successorBelief
    rw [← mul_assoc, mul_div_cancel₀ _ (hpos e)]
  calc ∑ e : Packet, Ze b K e * ∑ x' : World, successorBelief b K e (hpos e) x' * g x'
      = ∑ e : Packet, ∑ x' : World, jointP b K x' e * g x' :=
        Finset.sum_congr rfl fun e _ => per e
    _ = ∑ x' : World, ∑ e : Packet, jointP b K x' e * g x' := by
        rw [Finset.sum_comm]

/-- `packetMarginal` is nonneg (each `K.prob` is). -/
theorem packetMarginal_nonneg (K : TransKernel World Packet) (x : World) (e : Packet) :
    0 ≤ packetMarginal K x e := by
  unfold packetMarginal
  apply Finset.sum_nonneg; intro x' _
  exact K.nonneg x x' e

/-- Support propagation (finite exact-Bayes half of the particle-filter
resampling story): if some prior world has positive mass AND positive
marginal for `e`, the evidence is strictly positive. Particle analog: Pitt
& Shephard 1999 APF first-stage weights `g(k|Y) ∝ π^k·f(y|μ^k)` need
positive predictive mass to resample from
(https://en.wikipedia.org/wiki/Auxiliary_particle_filter). -/
theorem Ze_pos_of_packetMarginal_pos (b : Belief World) (K : TransKernel World Packet)
    (e : Packet) (h : ∃ x, 0 < b.prob x ∧ 0 < packetMarginal K x e) :
    0 < Ze b K e := by
  rw [Ze_eq_weighted_marginal]
  obtain ⟨x, hpx, hmx⟩ := h
  refine Finset.sum_pos' ?_ ?_
  · intro y _
    exact mul_nonneg (b.nonneg y) (packetMarginal_nonneg K y e)
  · exact ⟨x, Finset.mem_univ x, mul_pos hpx hmx⟩

/-- Persistence: a packaged posterior followed by a
strictly-supported kernel keeps `Ze` positive — so `successor_tower`'s
`∀ e, Ze ≠ 0` burden discharges for step two and updates iterate. The
full-support hypothesis is sufficient, not necessary (documented, not
hidden); weakening it to per-packet support is a stated follow-up. -/
theorem successor_Ze_pos_persistent (b : Belief World) (K1 K2 : TransKernel World Packet)
    (e1 e2 : Packet) (he1 : Ze b K1 e1 ≠ 0)
    (hsupp : ∀ x x', 0 < K2.prob x x' e2) :
    0 < Ze (successorBeliefPackaged b K1 e1 he1) K2 e2 := by
  apply Ze_pos_of_packetMarginal_pos
  have hsum : ∑ x' : World, (successorBeliefPackaged b K1 e1 he1).prob x' = 1 :=
    (successorBeliefPackaged b K1 e1 he1).sum_one
  have hne : ∃ x', (successorBeliefPackaged b K1 e1 he1).prob x' ≠ 0 := by
    by_contra hcon
    push Not at hcon
    have hzero : ∑ x' : World, (successorBeliefPackaged b K1 e1 he1).prob x' = 0 :=
      Finset.sum_eq_zero (fun x _ => hcon x)
    linarith
  obtain ⟨x', hx'⟩ := hne
  have hpos : 0 < (successorBeliefPackaged b K1 e1 he1).prob x' :=
    lt_of_le_of_ne' ((successorBeliefPackaged b K1 e1 he1).nonneg x') hx'
  refine ⟨x', hpos, ?_⟩
  unfold packetMarginal
  refine Finset.sum_pos' ?_ ?_
  · intro y _
    exact le_of_lt (hsupp x' y)
  · exact ⟨x', Finset.mem_univ x', hsupp x' x'⟩
/-- Packaged joint factors through step-one mass (algebraic core of
iterate-compose: two updates equal one joint step up to the `Ze` product;
mirrors `successor_tower`'s per-`e` `mul_div_cancel₀`). -/
theorem jointP_packaged_mul_Ze (b : Belief World) (K1 K2 : TransKernel World Packet)
    (e1 : Packet) (he1 : Ze b K1 e1 ≠ 0) (x'' : World) (e2 : Packet) :
    jointP (successorBeliefPackaged b K1 e1 he1) K2 x'' e2 * Ze b K1 e1
      = ∑ x' : World, jointP b K1 x' e1 * K2.prob x' x'' e2 := by
  have hJ : jointP (successorBeliefPackaged b K1 e1 he1) K2 x'' e2
      = ∑ x' : World, (jointP b K1 x' e1 / Ze b K1 e1) * K2.prob x' x'' e2 := rfl
  rw [hJ, Finset.sum_mul]
  refine Finset.sum_congr rfl fun x' _ => ?_
  rw [div_mul_eq_mul_div, div_mul_cancel₀ _ he1]

/-- Iterate equals two-step joint (pointwise): two packaged updates collapse
to one joint sum up to the `Ze` product. Closes the BeliefWin thread: tower
gives the marginal law, `jointP_packaged_mul_Ze` the joint core, this the
pointwise iterate equality — `successorBeliefPackaged` is genuinely
iterable. `he2` discharges via `successor_Ze_pos_persistent` + `ne_of_gt`
under full support. -/
theorem successor_iterate_eq_twoStep (b : Belief World) (K1 K2 : TransKernel World Packet)
    (e1 e2 : Packet) (he1 : Ze b K1 e1 ≠ 0)
    (he2 : Ze (successorBeliefPackaged b K1 e1 he1) K2 e2 ≠ 0) (x'' : World) :
    successorBelief (successorBeliefPackaged b K1 e1 he1) K2 e2 he2 x''
      = (∑ x' : World, jointP b K1 x' e1 * K2.prob x' x'' e2)
        / (Ze (successorBeliefPackaged b K1 e1 he1) K2 e2 * Ze b K1 e1) := by
  have hJ := jointP_packaged_mul_Ze b K1 K2 e1 he1 x'' e2
  have hJ2 : jointP (successorBeliefPackaged b K1 e1 he1) K2 x'' e2
      = (∑ x' : World, jointP b K1 x' e1 * K2.prob x' x'' e2) / Ze b K1 e1 :=
    (eq_div_iff he1).mpr hJ
  unfold successorBelief
  rw [hJ2, div_div, mul_comm (Ze b K1 e1)]

theorem reweight_without_pushforward_is_wrong :
    ∃ (World2 : Type) (_ : Fintype World2) (_ : DecidableEq World2)
      (Packet2 : Type) (_ : Fintype Packet2) (_ : DecidableEq Packet2)
      (b : Belief World2) (K : TransKernel World2 Packet2) (e : Packet2)
      (he : Ze b K e ≠ 0),
      (∃ g : World2 → ℝ, (∑ x : World2, (b.prob x * packetMarginal K x e / Ze b K e) * g x)
        ≠ (∑ x' : World2, successorBelief b K e he x' * g x')) := by
  refine ⟨Bool, inferInstance, inferInstance, Bool, inferInstance, inferInstance, ?_⟩
  -- Define Belief: non-uniform prior
  let b : Belief Bool :=
    { prob := fun w => if w then (0.25 : ℝ) else 0.75
      nonneg := by
        intro x
        cases x <;> simp <;> norm_num
      sum_one := by
        have h_pair : ∑ x : Bool, (if x then (0.25 : ℝ) else (0.75 : ℝ)) = (0.25 : ℝ) + 0.75 := by
          have h_univ : (Finset.univ : Finset Bool) = ({true, false} : Finset Bool) := by decide
          have h_ne : (true : Bool) ≠ false := by decide
          calc ∑ x : Bool, (if x then (0.25 : ℝ) else 0.75)
              = ∑ x ∈ ({true, false} : Finset Bool), (if x then (0.25 : ℝ) else 0.75) := by rw [← h_univ]
            _ = (if (true : Bool) then (0.25 : ℝ) else 0.75) + (if (false : Bool) then (0.25 : ℝ) else 0.75) := by rw [Finset.sum_pair h_ne]
            _ = (0.25 : ℝ) + 0.75 := by simp
        rw [h_pair]
        norm_num
    }
  -- Define swap kernel: deterministic swap on packet false
  let K : TransKernel Bool Bool :=
    { prob := fun x x' e => if x' = !x ∧ e = false then (1 : ℝ) else 0
      nonneg := by
        intro x x' e
        by_cases h : x' = !x ∧ e = false
        · simp [h]
        · simp [h]
      stochastic := by
        intro x
        -- Helper: sum over Bool is f true + f false
        have h_sum_bool : ∀ (f : Bool → ℝ), ∑ x' : Bool, f x' = f true + f false := by
          intro f
          have h_univ : (Finset.univ : Finset Bool) = ({true, false} : Finset Bool) := by decide
          have h_ne : (true : Bool) ≠ false := by decide
          calc ∑ x' : Bool, f x'
              = ∑ x' ∈ ({true, false} : Finset Bool), f x' := by rw [← h_univ]
            _ = f true + f false := by rw [Finset.sum_pair h_ne]
        have h_sum_bool2 : ∀ (f : Bool → ℝ), ∑ e : Bool, f e = f true + f false := h_sum_bool
        cases x with
        | true =>
          -- x = true, !x = false, so K true x' e = 1 iff x'=false ∧ e=false
          have h_inner_true : ∀ x' : Bool, ∑ e : Bool, (if x' = !true ∧ e = false then (1 : ℝ) else 0) = if x' = false then 1 else 0 := by
            intro x'
            rw [h_sum_bool2]
            cases x' with
            | true =>
              simp
            | false =>
              simp
          have h_outer : ∑ x' : Bool, ∑ e : Bool, (if x' = !true ∧ e = false then (1 : ℝ) else 0) = 1 := by
            rw [h_sum_bool]
            simp [h_inner_true]
          simp only [Bool.not_true] at h_outer ⊢
          exact h_outer
        | false =>
          have h_inner_false : ∀ x' : Bool, ∑ e : Bool, (if x' = !false ∧ e = false then (1 : ℝ) else 0) = if x' = true then 1 else 0 := by
            intro x'
            rw [h_sum_bool2]
            cases x' with
            | true =>
              simp
            | false =>
              simp
          have h_outer : ∑ x' : Bool, ∑ e : Bool, (if x' = !false ∧ e = false then (1 : ℝ) else 0) = 1 := by
            rw [h_sum_bool]
            simp [h_inner_false]
          simp only [Bool.not_false] at h_outer ⊢
          exact h_outer
    }
  let e : Bool := false
  have h_packet : ∀ x : Bool, packetMarginal K x e = 1 := by
    intro x
    unfold packetMarginal
    have h_sum_bool : ∀ (f : Bool → ℝ), ∑ x' : Bool, f x' = f true + f false := by
      intro f
      have h_univ : (Finset.univ : Finset Bool) = ({true, false} : Finset Bool) := by decide
      have h_ne : (true : Bool) ≠ false := by decide
      calc ∑ x' : Bool, f x'
          = ∑ x' ∈ ({true, false} : Finset Bool), f x' := by rw [← h_univ]
        _ = f true + f false := by rw [Finset.sum_pair h_ne]
    cases x with
    | true =>
      simp only [K, e, Bool.not_true]
      rw [h_sum_bool]
      simp
    | false =>
      simp only [K, e, Bool.not_false]
      rw [h_sum_bool]
      simp
  have h_Ze : Ze b K e = 1 := by
    have h_eq : Ze b K e = ∑ x : Bool, b.prob x * packetMarginal K x e := Ze_eq_weighted_marginal b K e
    rw [h_eq]
    have h_sum_bool : ∀ (f : Bool → ℝ), ∑ x : Bool, f x = f true + f false := by
      intro f
      have h_univ : (Finset.univ : Finset Bool) = ({true, false} : Finset Bool) := by decide
      have h_ne : (true : Bool) ≠ false := by decide
      calc ∑ x : Bool, f x
          = ∑ x ∈ ({true, false} : Finset Bool), f x := by rw [← h_univ]
        _ = f true + f false := by rw [Finset.sum_pair h_ne]
    simp_rw [h_packet]
    rw [h_sum_bool]
    simp only [b]
    norm_num
  have he : Ze b K e ≠ 0 := by rw [h_Ze]; norm_num
  refine ⟨b, K, e, he, ?_⟩
  let g : Bool → ℝ := fun w => if w then 1 else 0
  use g
  -- Compute left: ∑ x, (b.prob x * 1 /1) * g x = b.prob true =0.25
  have h_left : (∑ x : Bool, (b.prob x * packetMarginal K x e / Ze b K e) * g x) = 0.25 := by
    have h_sum_bool : ∀ (f : Bool → ℝ), ∑ x : Bool, f x = f true + f false := by
      intro f
      have h_univ : (Finset.univ : Finset Bool) = ({true, false} : Finset Bool) := by decide
      have h_ne : (true : Bool) ≠ false := by decide
      calc ∑ x : Bool, f x
          = ∑ x ∈ ({true, false} : Finset Bool), f x := by rw [← h_univ]
        _ = f true + f false := by rw [Finset.sum_pair h_ne]
    rw [h_sum_bool]
    simp only [g, b, h_packet, h_Ze]
    norm_num
  -- Compute jointP values
  have h_joint_true : jointP b K true e = 0.75 := by
    unfold jointP
    have h_sum_bool : ∀ (f : Bool → ℝ), ∑ x : Bool, f x = f true + f false := by
      intro f
      have h_univ : (Finset.univ : Finset Bool) = ({true, false} : Finset Bool) := by decide
      have h_ne : (true : Bool) ≠ false := by decide
      calc ∑ x : Bool, f x
          = ∑ x ∈ ({true, false} : Finset Bool), f x := by rw [← h_univ]
        _ = f true + f false := by rw [Finset.sum_pair h_ne]
    rw [h_sum_bool]
    simp only [b, K, e, Bool.not_true, Bool.not_false]
    norm_num
  have h_joint_false : jointP b K false e = 0.25 := by
    unfold jointP
    have h_sum_bool : ∀ (f : Bool → ℝ), ∑ x : Bool, f x = f true + f false := by
      intro f
      have h_univ : (Finset.univ : Finset Bool) = ({true, false} : Finset Bool) := by decide
      have h_ne : (true : Bool) ≠ false := by decide
      calc ∑ x : Bool, f x
          = ∑ x ∈ ({true, false} : Finset Bool), f x := by rw [← h_univ]
        _ = f true + f false := by rw [Finset.sum_pair h_ne]
    rw [h_sum_bool]
    simp only [b, K, e, Bool.not_true, Bool.not_false]
    norm_num
  have h_right : (∑ x' : Bool, successorBelief b K e he x' * g x') = 0.75 := by
    have h_sum_bool : ∀ (f : Bool → ℝ), ∑ x' : Bool, f x' = f true + f false := by
      intro f
      have h_univ : (Finset.univ : Finset Bool) = ({true, false} : Finset Bool) := by decide
      have h_ne : (true : Bool) ≠ false := by decide
      calc ∑ x' : Bool, f x'
          = ∑ x' ∈ ({true, false} : Finset Bool), f x' := by rw [← h_univ]
        _ = f true + f false := by rw [Finset.sum_pair h_ne]
    rw [h_sum_bool]
    unfold successorBelief
    rw [h_joint_true, h_joint_false, h_Ze]
    simp only [g]
    norm_num
  rw [h_left, h_right]
  norm_num
theorem missing_mass_detected (b : Belief World) (K : TransKernel World Packet) (PacketSubset : Finset Packet) (hMissing : ∃ e ∉ PacketSubset, Ze b K e > 0) : ∑ e ∈ PacketSubset, Ze b K e < 1 := by
  obtain ⟨e0, he0_not_mem, he0_pos⟩ := hMissing
  have total := packet_partition_Z_sum_one b K
  have add_le : ∑ e ∈ PacketSubset, Ze b K e + Ze b K e0 ≤ ∑ e : Packet, Ze b K e := by
    have hsub : PacketSubset ∪ {e0} ⊆ Finset.univ := by simp
    have hdisj : Disjoint PacketSubset {e0} := by rw [Finset.disjoint_singleton_right]; exact he0_not_mem
    have hunion_sum : ∑ e ∈ PacketSubset ∪ {e0}, Ze b K e = ∑ e ∈ PacketSubset, Ze b K e + Ze b K e0 := by rw [Finset.sum_union hdisj]; simp
    calc ∑ e ∈ PacketSubset, Ze b K e + Ze b K e0 = ∑ e ∈ PacketSubset ∪ {e0}, Ze b K e := hunion_sum.symm
      _ ≤ ∑ e : Packet, Ze b K e := Finset.sum_le_sum_of_subset_of_nonneg hsub (fun e _ _ => Ze_nonneg b K e)
  linarith

theorem duplicate_packet_would_overcount (b : Belief World) (K : TransKernel World Packet) (e1 e2 : Packet) (h12 : e1 ≠ e2) : Ze b K e1 + Ze b K e2 ≤ 1 := by
  have total := packet_partition_Z_sum_one b K
  calc Ze b K e1 + Ze b K e2 ≤ ∑ e : Packet, Ze b K e := by
        have hpair : ({e1, e2} : Finset Packet) ⊆ Finset.univ := by simp
        have hsum_pair : ∑ e ∈ ({e1, e2} : Finset Packet), Ze b K e = Ze b K e1 + Ze b K e2 := by rw [Finset.sum_pair h12]
        calc Ze b K e1 + Ze b K e2 = ∑ e ∈ ({e1, e2} : Finset Packet), Ze b K e := hsum_pair.symm
          _ ≤ ∑ e : Packet, Ze b K e := Finset.sum_le_sum_of_subset_of_nonneg hpair (fun e _ _ => Ze_nonneg b K e)
    _ = 1 := total

end SuccessorBelief

section InformationSet

variable {World : Type} [Fintype World] [DecidableEq World]
variable {Obs : Type} [DecidableEq Obs]
variable {Action : Type} [Fintype Action] [DecidableEq Action]

structure InfoPolicy (World Obs Action : Type) [Fintype World] [Fintype Action] where
  info : World → Obs
  policy : Obs → Action → ℝ
  policy_nonneg : ∀ o a, 0 ≤ policy o a
  policy_stochastic : ∀ o, ∑ a : Action, policy o a = 1

theorem info_invariant_same_obs_same_policy (P : InfoPolicy World Obs Action) (x x' : World) (h : P.info x = P.info x') (a : Action) : P.policy (P.info x) a = P.policy (P.info x') a := by rw [h]

theorem info_factorization_is_measurable (P : InfoPolicy World Obs Action) : ∀ x x', P.info x = P.info x' → ∀ a, P.policy (P.info x) a = P.policy (P.info x') a := by intros x x' h a; exact info_invariant_same_obs_same_policy P x x' h a

theorem info_leak_would_violate : ∃ (badPolicy : Bool → Fin 2 → ℝ), (∃ x x' : Bool, (fun _ : Bool => (0 : Unit)) x = (fun _ => (0:Unit)) x' ∧ badPolicy x ≠ badPolicy x') := by
  refine ⟨fun w => fun a => if w then (if a=0 then 1 else 0) else (if a=0 then 0 else 1), ?_⟩
  use true, false
  constructor
  · rfl
  · intro h
    have eq0 := congr_fun h 0
    simp at eq0

end InformationSet

section Confirmation

variable {World : Type} [Fintype World] [DecidableEq World]
variable {Outcome : Type} [Fintype Outcome] [DecidableEq Outcome]

noncomputable def confirmationDelta (prob : World → ℝ) (kernels : Fin 2 → World → Outcome → ℝ) (utility : Outcome → Hydra2.Blueprint.Objective.UtilityVector) (seat : Hydra2.Blueprint.Objective.Seat) : ℝ :=
  ∑ x : World, prob x * ((∑ o : Outcome, kernels 0 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) - (∑ o : Outcome, kernels 1 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat))

theorem confirmation_estimator_unbiased_for_delta (prob : World → ℝ) (kernels : Fin 2 → World → Outcome → ℝ) (utility : Outcome → Hydra2.Blueprint.Objective.UtilityVector) (seat : Hydra2.Blueprint.Objective.Seat) :
    confirmationDelta prob kernels utility seat =
    (∑ x : World, ∑ o : Outcome, prob x * kernels 0 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) - (∑ x : World, ∑ o : Outcome, prob x * kernels 1 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) := by
  unfold confirmationDelta
  have expand : ∀ x : World,
      prob x * ((∑ o : Outcome, kernels 0 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) - (∑ o : Outcome, kernels 1 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat))
      = (∑ o : Outcome, prob x * kernels 0 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) - (∑ o : Outcome, prob x * kernels 1 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) := by
    intro x; rw [mul_sub, Finset.mul_sum, Finset.mul_sum]; congr 1 <;> (congr 1; ext o; ring)
  simp_rw [expand, Finset.sum_sub_distrib]

end Confirmation

end Hydra2.Blueprint.Belief
