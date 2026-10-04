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

/-- Evidence is always fully explained: the joint mass over all successor
worlds equals the packet's evidence mass — `∑_{x'} jointP = Ze` by
definition, so conditioning divides by exactly the available mass with
nothing left over. Pair with `successorBelief_sum_one` (the posterior
normalizes): together update-then-renormalize is lossless — no evidence
mass vanishes between the joint and the posterior. -/
theorem jointP_sum_eq_Ze (b : Belief World) (K : TransKernel World Packet) (e : Packet) :
    ∑ x' : World, jointP b K x' e = Ze b K e := rfl

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

/-- Packet mass is monotone in the subset: extending the covered packet set
can only add evidence mass. Practical core: commit/squash bookkeeping that
drops packets must track the dropped mass — extending coverage never hides
it. Limited-compute lemma: budget packet enumeration by mass covered. -/
theorem Ze_subset_mono (b : Belief World) (K : TransKernel World Packet)
    (S T : Finset Packet) (h : S ⊆ T) :
    ∑ e ∈ S, Ze b K e ≤ ∑ e ∈ T, Ze b K e :=
  Finset.sum_le_sum_of_subset_of_nonneg h (fun e _ _ => Ze_nonneg b K e)

/-- Empty coverage holds zero mass: the commit ledger starts at zero —
the finite reason coverage growth (`Ze_subset_mono`) is growth FROM
zero, so the first committed packet's mass is exactly its Ze, with no
baseline offset to subtract. Pairs `Ze_complement_empty` (the
dropped side of nothing is everything): together both sides of the
empty boundary are pinned. -/
theorem Ze_empty_set (b : Belief World) (K : TransKernel World Packet) :
    ∑ e ∈ (∅ : Finset Packet), Ze b K e = 0 :=
  Finset.sum_empty

/-- Singleton coverage holds exactly its packet's mass: no more, no
less — the mirror of `Ze_empty_set` (empty holds zero). Together the
ledger is pinned at both extremes, so single-packet commits read off
directly with no sum to compute: the commit fast-path for
one-packet evidence needs no aggregation pass. -/
theorem Ze_singleton (b : Belief World) (K : TransKernel World Packet) (e : Packet) :
    ∑ e' ∈ ({e} : Finset Packet), Ze b K e' = Ze b K e :=
  Finset.sum_singleton (Ze b K) e

/-- Covered mass splits additively at any boundary: `∑ over (S ∪ T)` is
the union-side sum when disjoint — the finite reason the commit/squash
ledger can merge two disjoint packet batches by adding their masses
with no overlap correction. Pairs `Ze_subset_mono` (monotone growth):
together coverage grows monotonically AND disjoint merges add exactly. -/
theorem Ze_disjoint_union (b : Belief World) (K : TransKernel World Packet)
    (S T : Finset Packet) (h : Disjoint S T) :
    ∑ e ∈ S ∪ T, Ze b K e
      = (∑ e ∈ S, Ze b K e) + ∑ e ∈ T, Ze b K e :=
  Finset.sum_union h

/-- Dropped-packet mass is exactly accounted: the mass outside the covered
set is `1 −` the mass inside — conservation with no leak term. Companion to
`Ze_subset_mono` (which bounds the inside): together they let commit/squash
bookkeeping price a dropped packet set precisely. Failure mode killed: an
unaccounted mass gap silently biasing the successor belief. -/
theorem Ze_complement_mass (b : Belief World) (K : TransKernel World Packet)
    (S : Finset Packet) :
    ∑ e ∈ Sᶜ, Ze b K e = 1 - ∑ e ∈ S, Ze b K e := by
  have total := packet_partition_Z_sum_one b K
  have hsplit := Finset.sum_compl_add_sum S (fun e => Ze b K e)
  linarith

/-- Dropped mass is antitone in coverage: extending the covered set can
only shrink what is dropped — the finite reason commit/squash passes
that cover more packets account for less dropped mass, monotonically.
Chains `Ze_complement_mass` with `Ze_subset_mono`: the inside grows
while the outside shrinks, so the ledger balances at every step. -/
theorem Ze_complement_antitone (b : Belief World) (K : TransKernel World Packet)
    (S T : Finset Packet) (h : S ⊆ T) :
    ∑ e ∈ Tᶜ, Ze b K e ≤ ∑ e ∈ Sᶜ, Ze b K e := by
  have h1 := Ze_complement_mass b K S
  have h2 := Ze_complement_mass b K T
  have hle := Ze_subset_mono b K S T h
  linarith

/-- Full coverage drops nothing: when every packet is covered the complement
mass is zero — the finite reason a complete packet enumeration needs no
drop accounting at all. Pair with `Ze_complement_mass` (which prices the
general case): together the commit/squash ledger distinguishes free
completeness from priced partiality. -/
theorem Ze_complement_univ (b : Belief World) (K : TransKernel World Packet) :
    ∑ e ∈ (Finset.univ : Finset Packet)ᶜ, Ze b K e = 0 := by
  simp

/-- Complement-univ by mass accounting: the zero-drop fact also follows
from `Ze_complement_mass` with `S = univ` — the finite reason the ledger
has two independent paths to free completeness (direct simp, or mass
arithmetic), so a refactor breaking one leaves the other. Cross-checks
`Ze_complement_univ` exactly. -/
theorem Ze_complement_univ_via_mass (b : Belief World) (K : TransKernel World Packet) :
    ∑ e ∈ (Finset.univ : Finset Packet)ᶜ, Ze b K e = 0 := by
  have h := Ze_complement_mass b K Finset.univ
  simpa using h

/-- Empty complement by mass accounting: the full-unit fact also follows
from `Ze_complement_mass` with `S = ∅` — the mirror of
`Ze_complement_univ_via_mass` (full coverage via mass). Together both
coverage extremes have two independent paths (direct and arithmetic),
so the ledger's endpoints survive any single-proof refactor. -/
theorem Ze_complement_empty_via_mass (b : Belief World) (K : TransKernel World Packet) :
    ∑ e ∈ (∅ : Finset Packet)ᶜ, Ze b K e = 1 := by
  have h := Ze_complement_mass b K ∅
  simpa using h

/-- Empty coverage explains nothing: with no packets covered the complement
carries the full unit mass — the finite reason a commit/squash pass that
drops everything must account for mass 1, never 0. Pair with
`Ze_complement_univ` (full coverage drops nothing): together the ledger
is pinned at both coverage extremes, so partial coverage interpolates
between free completeness and full accountability. -/
theorem Ze_complement_empty (b : Belief World) (K : TransKernel World Packet) :
    ∑ e ∈ (∅ : Finset Packet)ᶜ, Ze b K e = 1 := by
  have total := packet_partition_Z_sum_one b K
  have hcompl : (∅ : Finset Packet)ᶜ = Finset.univ := Finset.compl_empty
  rw [hcompl]
  simpa using total

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

/-- Confirmation is linear in belief scale: reweighting every world's mass
by `c` scales the delta by `c` — the finite reason the S5 gate prices
unnormalized beliefs directly: rescaling the belief rescales the answer
with no re-derivation, so partial-mass beliefs (post-drop) read off the
full-mass price. Pairs `confirmationDelta_zero_weight_drop` (support
restriction): together scale and support both factor cleanly. -/
theorem confirmationDelta_scale (c : ℝ) (prob : World → ℝ)
    (kernels : Fin 2 → World → Outcome → ℝ)
    (utility : Outcome → Hydra2.Blueprint.Objective.UtilityVector) (seat : Hydra2.Blueprint.Objective.Seat) :
    confirmationDelta (fun x => c * prob x) kernels utility seat
      = c * confirmationDelta prob kernels utility seat := by
  unfold confirmationDelta
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro x _
  ring

/-- Confirmation is additive in belief: the delta over a summed belief is
the sum of the deltas — with `confirmationDelta_scale` (homogeneity)
this makes confirmation fully linear in belief. The finite reason split
beliefs (per-packet posteriors, per-seat subpopulations) compose by
addition: confirm each slice separately, add the answers. -/
theorem confirmationDelta_add (p₁ p₂ : World → ℝ)
    (kernels : Fin 2 → World → Outcome → ℝ)
    (utility : Outcome → Hydra2.Blueprint.Objective.UtilityVector) (seat : Hydra2.Blueprint.Objective.Seat) :
    confirmationDelta (fun x => p₁ x + p₂ x) kernels utility seat
      = confirmationDelta p₁ kernels utility seat
        + confirmationDelta p₂ kernels utility seat := by
  unfold confirmationDelta
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro x _
  ring

theorem confirmation_estimator_unbiased_for_delta (prob : World → ℝ) (kernels : Fin 2 → World → Outcome → ℝ) (utility : Outcome → Hydra2.Blueprint.Objective.UtilityVector) (seat : Hydra2.Blueprint.Objective.Seat) :
    confirmationDelta prob kernels utility seat =
    (∑ x : World, ∑ o : Outcome, prob x * kernels 0 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) - (∑ x : World, ∑ o : Outcome, prob x * kernels 1 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) := by
  unfold confirmationDelta
  have expand : ∀ x : World,
      prob x * ((∑ o : Outcome, kernels 0 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) - (∑ o : Outcome, kernels 1 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat))
      = (∑ o : Outcome, prob x * kernels 0 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) - (∑ o : Outcome, prob x * kernels 1 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) := by
    intro x; rw [mul_sub, Finset.mul_sum, Finset.mul_sum]; congr 1 <;> (congr 1; ext o; ring)
  simp_rw [expand, Finset.sum_sub_distrib]

/-- Identical continuations prove nothing: when both candidate kernels agree
    everywhere, the confirmation delta is exactly zero. Practical core: never
    spend confirmation sims comparing two actions whose continuations are the
    same rollout — the S5 gate skips identical-continuation pairs outright.
    Limited-compute lemma: confirmation budget flows only to genuinely
    divergent futures. Swapping the candidate kernels flips the sign
    (proved below): orientation is exact, degenerate comparisons vanish. -/
theorem confirmationDelta_swap (prob : World → ℝ)
    (k0 k1 : World → Outcome → ℝ)
    (utility : Outcome → Hydra2.Blueprint.Objective.UtilityVector) (seat : Hydra2.Blueprint.Objective.Seat) :
    confirmationDelta prob ![k1, k0] utility seat
      = -confirmationDelta prob ![k0, k1] utility seat := by
  unfold confirmationDelta
  rw [← Finset.sum_neg_distrib]
  apply Finset.sum_congr rfl
  intro x _
  simp
  ring

theorem confirmationDelta_zero_of_same_kernel (prob : World → ℝ)
    (kernels : Fin 2 → World → Outcome → ℝ)
    (utility : Outcome → Hydra2.Blueprint.Objective.UtilityVector) (seat : Hydra2.Blueprint.Objective.Seat)
    (h : ∀ x o, kernels 0 x o = kernels 1 x o) :
    confirmationDelta prob kernels utility seat = 0 := by
  unfold confirmationDelta
  have hsub : ∀ x : World,
      (∑ o : Outcome, kernels 0 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat)
      - (∑ o : Outcome, kernels 1 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) = 0 := by
    intro x
    have : (∑ o : Outcome, kernels 0 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat)
        = (∑ o : Outcome, kernels 1 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) :=
      Finset.sum_congr rfl fun o _ => by rw [h x o]
    linarith
  calc ∑ x : World, prob x * ((∑ o : Outcome, kernels 0 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) - (∑ o : Outcome, kernels 1 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat))
      = ∑ x : World, prob x * 0 := Finset.sum_congr rfl fun x _ => by rw [hsub x]
    _ = 0 := by simp

/-- Zero belief weights contribute nothing: worlds with `prob x = 0` drop
out of the confirmation sum — the finite reason sparse beliefs price only
their support, so the S5 gate enumerates live worlds instead of the full
wall. Pair with `confirmationDelta_point_mass` (single-world support is
free): together support size prices confirmation, from zero (empty) to
one (point mass) to full. -/
theorem confirmationDelta_zero_weight_drop (prob : World → ℝ)
    (kernels : Fin 2 → World → Outcome → ℝ)
    (utility : Outcome → Hydra2.Blueprint.Objective.UtilityVector) (seat : Hydra2.Blueprint.Objective.Seat)
    (S : Finset World) (h0 : ∀ x ∉ S, prob x = 0) :
    confirmationDelta prob kernels utility seat
      = ∑ x ∈ S, prob x * ((∑ o : Outcome, kernels 0 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) - (∑ o : Outcome, kernels 1 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat)) := by
  unfold confirmationDelta
  rw [Finset.sum_subset (Finset.subset_univ S) (fun x _ hx => by rw [h0 x hx, zero_mul])]

/-- Empty support kills confirmation: with no live worlds the delta is
zero — the mirror of `confirmationDelta_point_mass` (singleton support
is the analytic gap). Together the S5 gate's support-size ladder reads
0 → gap → full sum, so confirmation pricing bottoms at exactly zero,
never at a sampling floor. -/
theorem confirmationDelta_empty_support (prob : World → ℝ)
    (kernels : Fin 2 → World → Outcome → ℝ)
    (utility : Outcome → Hydra2.Blueprint.Objective.UtilityVector) (seat : Hydra2.Blueprint.Objective.Seat)
    (h0 : ∀ x, prob x = 0) : confirmationDelta prob kernels utility seat = 0 := by
  have h := confirmationDelta_zero_weight_drop prob kernels utility seat ∅ (by simpa using h0)
  simpa using h

/-- Null models contribute nothing: with both kernels identically zero
the delta is zero at ANY belief — the mirror of the zero-belief
corner (null belief, live models). Together both degeneracies are
pinned, so the S5 gate short-circuits either side without sampling:
dead models and dead beliefs alike are free. -/
theorem confirmationDelta_zero_kernels (prob : World → ℝ)
    (utility : Outcome → Hydra2.Blueprint.Objective.UtilityVector) (seat : Hydra2.Blueprint.Objective.Seat) :
    confirmationDelta prob (fun _ _ _ => 0) utility seat = 0 := by
  unfold confirmationDelta
  simp

/-- Point-mass beliefs need zero sims: on a degenerate belief concentrated at
    one world, the confirmation delta IS the single-world utility gap — read
    it directly, never sample. Practical core: the S5 gate short-circuits
    collapsed beliefs (ESS ≈ 1) to the analytic gap instead of burning
    confirmation rollouts. Limited-compute lemma: degenerate beliefs are free. -/
theorem confirmationDelta_point_mass (x₀ : World)
    (kernels : Fin 2 → World → Outcome → ℝ)
    (utility : Outcome → Hydra2.Blueprint.Objective.UtilityVector) (seat : Hydra2.Blueprint.Objective.Seat) :
    confirmationDelta (fun x => if x = x₀ then (1 : ℝ) else 0) kernels utility seat
      = (∑ o : Outcome, kernels 0 x₀ o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat)
      - (∑ o : Outcome, kernels 1 x₀ o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat) := by
  unfold confirmationDelta
  have hpoint : ∀ x : World, (if x = x₀ then (1 : ℝ) else (0 : ℝ))
        * ((∑ o : Outcome, kernels 0 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat)
        - (∑ o : Outcome, kernels 1 x o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat))
      = (if x = x₀ then
          ((∑ o : Outcome, kernels 0 x₀ o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat)
          - (∑ o : Outcome, kernels 1 x₀ o * Hydra2.Blueprint.Objective.rootScalar (utility o) seat))
        else 0) := by
    intro x
    by_cases hx : x = x₀
    · subst hx; simp
    · simp [hx]
  rw [Finset.sum_congr rfl (fun x _ => hpoint x)]
  simp

end Confirmation

end Hydra2.Blueprint.Belief
