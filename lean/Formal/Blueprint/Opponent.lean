import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Log.Basic

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedDecidableInType false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false

/-!
# Hydra2 §15 Candidate 8 — Joint Type/World Correlation

Blueprint §15: `p_next(θ,x') ∝ ∫ p_h(θ,x) K_h(dx',e|x,q_j(·|I_j(x),θ))`
and `Q_set` uncertainty set over coherent information-set policies.

Key non-factorization: updating only `p(θ)` against a type-independent `b(x)` loses
the induced `θ–x` correlation and can double-condition on `e`. The joint must be
maintained: `p(θ,x) → p_next(θ,x') → p_next(θ)=∫p_next`, `b_next(x'|θ)=p_next/p_next`.

For editors: this file uses finite sums to model the tiny-oracle joint posterior.
`θ` is opponent type, `x` is hidden world, `e` is observed packet, `K` includes
`q_j` likelihood + physical transition + call/pass resolution.
-/

namespace Hydra2.Blueprint.Opponent

section JointPosterior

variable {Theta World : Type} [Fintype Theta] [DecidableEq Theta] [Fintype World] [DecidableEq World]

/-- Joint law `p_h(θ,x)`. -/
structure JointThetaWorld (Theta World : Type) [Fintype Theta] [Fintype World] where
  prob : Theta → World → ℝ
  nonneg : ∀ t w, 0 ≤ prob t w
  sum_one : ∑ t : Theta, ∑ w : World, prob t w = 1

/-- Kernel `K_h(dx',e|x,θ)` — here simplified to `K x' e x t` for the observed `e`. -/
noncomputable def nextJoint
    (p : JointThetaWorld Theta World)
    (K : World → World → Theta → ℝ) -- `K(x',e|x,t)` for fixed `e`
    : Theta → World → ℝ :=
  fun t x' => ∑ x : World, p.prob t x * K x x' t

noncomputable def nextMarginalTheta
    (p : JointThetaWorld Theta World)
    (K : World → World → Theta → ℝ)
    (t : Theta) : ℝ :=
  ∑ x' : World, nextJoint p K t x'

/-- Conditional `b_next(x'|θ)=p_next(θ,x')/p_next(θ)` when `p_next(θ)>0`. -/
noncomputable def condWorldGivenTheta
    (p : JointThetaWorld Theta World)
    (K : World → World → Theta → ℝ)
    (t : Theta) (hne : nextMarginalTheta p K t ≠ 0) (x' : World) : ℝ :=
  nextJoint p K t x' / nextMarginalTheta p K t

theorem condWorldGivenTheta_nonneg
    (p : JointThetaWorld Theta World) (K : World → World → Theta → ℝ)
    (t : Theta) (hne : nextMarginalTheta p K t ≠ 0) (x' : World)
    (hK_nonneg : ∀ x x' t, 0 ≤ K x x' t) :
    0 ≤ condWorldGivenTheta p K t hne x' := by
  unfold condWorldGivenTheta nextJoint
  apply div_nonneg
  · apply Finset.sum_nonneg; intro x _; exact mul_nonneg (p.nonneg t x) (hK_nonneg _ _ _)
  · -- `nextMarginal ≥0` because it's sum of nonnegs, and `≠0` ⇒ `>0`
    have hnn : 0 ≤ nextMarginalTheta p K t := by
      unfold nextMarginalTheta nextJoint
      apply Finset.sum_nonneg; intro x' _; apply Finset.sum_nonneg
      intro x _; exact mul_nonneg (p.nonneg t x) (hK_nonneg _ _ _)
    linarith

theorem condWorldGivenTheta_sum_one
    (p : JointThetaWorld Theta World) (K : World → World → Theta → ℝ)
    (t : Theta) (hne : nextMarginalTheta p K t ≠ 0) :
    ∑ x' : World, condWorldGivenTheta p K t hne x' = 1 := by
  unfold condWorldGivenTheta
  have h : (∑ x' : World, nextJoint p K t x') = nextMarginalTheta p K t := rfl
  rw [← Finset.sum_div, h]
  exact div_self hne

/-- Type-posterior Bayes update (BPR Eq.1-2, via BprVog scout, arXiv:1505.00284 §2.9: `β^t(τ) = P(σ^t|τ,π^t)β^{t-1}(τ) / Σ_{τ'} P(σ^t|τ',π^t)β^{t-1}(τ')`, i.e. `η·F(τ)·β(τ)` with observation model `F` per Def.7). Dual of `condWorldGivenTheta` (which conditions worlds on a type; this conditions types on an observed packet via likelihood `L(τ) = q_j`-packet likelihood). Posterior sums to one whenever the evidence is nonzero — the finite core behind Hydra2's two-level posterior (θ over strategies, signal packet as observation). Full Dirichlet-multinomial + BPR policy-selection (PI/EI/BE/KG) stays backend-side. -/
noncomputable def typePosterior
    (beta L : Theta → ℝ) : Theta → ℝ :=
  fun t => L t * beta t / ∑ t' : Theta, L t' * beta t'

theorem typePosterior_sum_one
    (beta L : Theta → ℝ)
    (hne : (∑ t' : Theta, L t' * beta t') ≠ 0) :
    ∑ t : Theta, typePosterior (Theta:=Theta) beta L t = 1 := by
  unfold typePosterior
  rw [← Finset.sum_div]
  exact div_self hne

/-- Non-factorization: `p_next(θ,x') ≠ p_next(θ)·b(x')` in general because `x` correlates
`θ` and the world via the joint and `K`. The marginal-only update `p_next(θ) ∝ p(θ)·something`
that ignores `b(x|θ)` double-conditions. We state existence of a witness. -/
theorem joint_does_not_factorize :
    ∃ (Theta0 : Type) (_ : Fintype Theta0) (_ : DecidableEq Theta0)
      (World0 : Type) (_ : Fintype World0) (_ : DecidableEq World0)
      (p : JointThetaWorld Theta0 World0) (K : World0 → World0 → Theta0 → ℝ),
      (∃ t x', nextJoint p K t x' ≠ nextMarginalTheta p K t * (∑ t2 : Theta0, nextJoint p K t2 x' / ∑ t3 : Theta0, nextMarginalTheta p K t3)) := by
  -- witness Bool×Bool uniform p, K deterministic on t
  let pProb : Bool → Bool → ℝ := fun _ _ => (1/4 : ℝ)
  let K : Bool → Bool → Bool → ℝ := fun _ x' t => if t = x' then (1 : ℝ) else (0 : ℝ)
  have hP_nonneg : ∀ t w : Bool, 0 ≤ pProb t w := by intro t w; unfold pProb; norm_num
  have hP_sum : (∑ t : Bool, ∑ w : Bool, pProb t w) = 1 := by
    have h_univ : (Finset.univ : Finset Bool) = {false, true} := by decide
    have h_inner : ∀ t : Bool, (∑ w : Bool, pProb t w) = (1/2 : ℝ) := by
      intro t
      rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
      unfold pProb; norm_num
    have h_outer : (∑ t : Bool, (1/2 : ℝ)) = (1 : ℝ) := by
      rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]; norm_num
    calc (∑ t : Bool, ∑ w : Bool, pProb t w) = ∑ t : Bool, (1/2 : ℝ) := by
          apply Finset.sum_congr rfl; intro t _; exact h_inner t
      _ = 1 := h_outer
  let p : JointThetaWorld Bool Bool := ⟨pProb, hP_nonneg, hP_sum⟩
  refine ⟨Bool, inferInstance, inferInstance, Bool, inferInstance, inferInstance, p, K, ?_⟩
  use true, true
  -- compute concrete values
  have h_univ : (Finset.univ : Finset Bool) = {false, true} := by decide
  have h_next_true_true : nextJoint (Theta:=Bool) (World:=Bool) p K true true = (1/2 : ℝ) := by
    unfold nextJoint p pProb K
    rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]
    simp [K]
    norm_num
  have h_next_true_false : nextJoint (Theta:=Bool) (World:=Bool) p K true false = (0 : ℝ) := by
    unfold nextJoint p pProb K
    rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]
    simp [K]
  have h_next_false_true : nextJoint (Theta:=Bool) (World:=Bool) p K false true = (0 : ℝ) := by
    unfold nextJoint p pProb K
    rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]
    simp [K]
  have h_next_false_false : nextJoint (Theta:=Bool) (World:=Bool) p K false false = (1/2 : ℝ) := by
    unfold nextJoint p pProb K
    rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]
    simp [K]
    norm_num
  have h_marg_true : nextMarginalTheta (Theta:=Bool) (World:=Bool) p K true = (1/2 : ℝ) := by
    unfold nextMarginalTheta
    rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]
    rw [h_next_true_true, h_next_true_false]
    norm_num
  have h_marg_false : nextMarginalTheta (Theta:=Bool) (World:=Bool) p K false = (1/2 : ℝ) := by
    unfold nextMarginalTheta
    rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]
    rw [h_next_false_true, h_next_false_false]
    norm_num
  have h_sum_next_true : (∑ t2 : Bool, nextJoint p K t2 true) = (1/2 : ℝ) := by
    rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]
    rw [h_next_true_true, h_next_false_true]
    norm_num
  have h_sum_marg : (∑ t3 : Bool, nextMarginalTheta p K t3) = (1 : ℝ) := by
    rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]
    rw [h_marg_true, h_marg_false]
    norm_num
  have h_rhs : nextMarginalTheta p K true * (∑ t2 : Bool, nextJoint p K t2 true / (∑ t3 : Bool, nextMarginalTheta p K t3)) = (1/4 : ℝ) := by
    have h_sum_div : (∑ t2 : Bool, nextJoint p K t2 true / (∑ t3 : Bool, nextMarginalTheta p K t3)) =
        (∑ t2 : Bool, nextJoint p K t2 true) / (∑ t3 : Bool, nextMarginalTheta p K t3) := by
      rw [Finset.sum_div]
    rw [h_sum_div, h_marg_true, h_sum_next_true, h_sum_marg]
    norm_num
  have h_goal : nextJoint p K true true ≠ nextMarginalTheta p K true * (∑ t2 : Bool, nextJoint p K t2 true / (∑ t3 : Bool, nextMarginalTheta p K t3)) := by
    rw [h_next_true_true, h_rhs]
    norm_num
  exact h_goal
/-- One-step update preserves total mass when the kernel is stochastic in `x'`:
`∑_{t,x'} nextJoint = 1` (swap sums via `sum_comm`, factor `K`-sums to `1`).
Finite contract behind sequential updates; correlation-across-time preservation
(`p_{t+1}` needs the full joint) stays HARD-skipped `MeasureTheory`. -/
theorem sequential_preserves_total_mass
    (p : JointThetaWorld Theta World) (K : World → World → Theta → ℝ)
    (hK_stoch : ∀ t x, ∑ x' : World, K x x' t = 1) :
    (∑ t : Theta, ∑ x' : World, nextJoint p K t x') = 1 := by
  have hstep : ∀ (t : Theta),
      (∑ x' : World, nextJoint p K t x') = ∑ x : World, p.prob t x := by
    intro t
    unfold nextJoint
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro x _
    rw [← Finset.mul_sum, hK_stoch, mul_one]
  calc (∑ t : Theta, ∑ x' : World, nextJoint p K t x')
      = ∑ t : Theta, ∑ x : World, p.prob t x := Finset.sum_congr rfl (fun t _ => hstep t)
    _ = 1 := p.sum_one
/-- One-step update packaged as a lawful joint law, so sequential updates iterate
without leaving `JointThetaWorld` (needs `K` nonneg + stochastic in `x'`). -/
noncomputable def nextJointLaw
    (p : JointThetaWorld Theta World) (K : World → World → Theta → ℝ)
    (hK_nonneg : ∀ x x' t, 0 ≤ K x x' t)
    (hK_stoch : ∀ t x, ∑ x' : World, K x x' t = 1) : JointThetaWorld Theta World where
  prob := nextJoint p K
  nonneg := fun t x' => Finset.sum_nonneg
    (fun x _ => mul_nonneg (p.nonneg t x) (hK_nonneg _ _ _))
  sum_one := sequential_preserves_total_mass p K hK_stoch

/-- Two-step mass iteration: total mass survives two updates chained through the
lawful packaging (step 1 inside `nextJointLaw`, step 2 by mass-preservation on
the packaged law). -/
theorem nextJointLaw_iterates_mass (p : JointThetaWorld Theta World)
    (K1 K2 : World → World → Theta → ℝ)
    (hN1 : ∀ x x' t, 0 ≤ K1 x x' t) (hS1 : ∀ t x, ∑ x' : World, K1 x x' t = 1)
    (hS2 : ∀ t x, ∑ x' : World, K2 x x' t = 1) :
    (∑ t : Theta, ∑ x'' : World, nextJoint (nextJointLaw p K1 hN1 hS1) K2 t x'') = 1 :=
  sequential_preserves_total_mass (nextJointLaw p K1 hN1 hS1) K2 hS2

/-- Two-step correlation persists: after a correlating update `K1` (deterministic
on `θ`, as in `joint_does_not_factorize`) followed by the identity kernel `K2`,
the joint still does not factorize. The second step copies the first-step joint
(`K2` is the identity, so `nextJoint` preserves the diagonal `1/2` law), hence
`LHS = 1/2 ≠ 1/4 = RHS` exactly as in the one-step witness. -/
theorem correlation_preserved_example :
    ∃ (p : JointThetaWorld Bool Bool) (K1 K2 : Bool → Bool → Bool → ℝ)
      (hN1 : ∀ x x' t, 0 ≤ K1 x x' t) (hS1 : ∀ t x, ∑ x' : Bool, K1 x x' t = 1),
      ∃ (t : Bool) (x'' : Bool),
        nextJoint (nextJointLaw p K1 hN1 hS1) K2 t x'' ≠
          nextMarginalTheta (nextJointLaw p K1 hN1 hS1) K2 t *
            (∑ t2 : Bool, nextJoint (nextJointLaw p K1 hN1 hS1) K2 t2 x'' /
              ∑ t3 : Bool, nextMarginalTheta (nextJointLaw p K1 hN1 hS1) K2 t3) := by
  let pProb : Bool → Bool → ℝ := fun _ _ => (1/4 : ℝ)
  let K1 : Bool → Bool → Bool → ℝ := fun _ x' t => if t = x' then (1 : ℝ) else (0 : ℝ)
  let K2 : Bool → Bool → Bool → ℝ := fun x x'' _ => if x = x'' then (1 : ℝ) else (0 : ℝ)
  have h_univ : (Finset.univ : Finset Bool) = {false, true} := by decide
  have hP_nonneg : ∀ t w : Bool, 0 ≤ pProb t w := by
    intro t w
    unfold pProb
    norm_num
  have hP_sum : (∑ t : Bool, ∑ w : Bool, pProb t w) = 1 := by
    have h_inner : ∀ t : Bool, (∑ w : Bool, pProb t w) = (1/2 : ℝ) := by
      intro t
      rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
      unfold pProb
      norm_num
    have h_outer : (∑ t : Bool, (1/2 : ℝ)) = (1 : ℝ) := by
      rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]
      norm_num
    calc (∑ t : Bool, ∑ w : Bool, pProb t w) = ∑ t : Bool, (1/2 : ℝ) := by
          apply Finset.sum_congr rfl; intro t _; exact h_inner t
      _ = 1 := h_outer
  have hN1 : ∀ x x' (t : Bool), 0 ≤ K1 x x' t := by
    intro x x' t
    unfold K1
    split <;> norm_num
  have hS1 : ∀ (t : Bool) (x : Bool), ∑ x' : Bool, K1 x x' t = 1 := by
    intro t x
    rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
    cases t <;> unfold K1 <;> simp
  let p : JointThetaWorld Bool Bool := ⟨pProb, hP_nonneg, hP_sum⟩
  refine ⟨p, K1, K2, hN1, hS1, true, true, ?_⟩
  have h1_tt : nextJoint (Theta:=Bool) (World:=Bool) p K1 true true = (1/2 : ℝ) := by
    unfold nextJoint p pProb K1
    rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
    simp
    norm_num
  have h1_tf : nextJoint (Theta:=Bool) (World:=Bool) p K1 true false = (0 : ℝ) := by
    unfold nextJoint p pProb K1
    rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
    simp
  have h1_ft : nextJoint (Theta:=Bool) (World:=Bool) p K1 false true = (0 : ℝ) := by
    unfold nextJoint p pProb K1
    rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
    simp
  have h1_ff : nextJoint (Theta:=Bool) (World:=Bool) p K1 false false = (1/2 : ℝ) := by
    unfold nextJoint p pProb K1
    rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
    simp
    norm_num
  have h2_tt : nextJoint (nextJointLaw p K1 hN1 hS1) K2 true true = (1/2 : ℝ) := by
    have e_tf : (nextJointLaw p K1 hN1 hS1).prob true false = 0 := h1_tf
    have e_tt : (nextJointLaw p K1 hN1 hS1).prob true true = (1/2 : ℝ) := h1_tt
    unfold nextJoint
    rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
    rw [e_tf, e_tt]
    unfold K2
    simp
  have h2_tf : nextJoint (nextJointLaw p K1 hN1 hS1) K2 true false = (0 : ℝ) := by
    have e_tf : (nextJointLaw p K1 hN1 hS1).prob true false = 0 := h1_tf
    have e_tt : (nextJointLaw p K1 hN1 hS1).prob true true = (1/2 : ℝ) := h1_tt
    unfold nextJoint
    rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
    rw [e_tf, e_tt]
    unfold K2
    simp
  have h2_ft : nextJoint (nextJointLaw p K1 hN1 hS1) K2 false true = (0 : ℝ) := by
    have e_ft : (nextJointLaw p K1 hN1 hS1).prob false true = 0 := h1_ft
    have e_ff : (nextJointLaw p K1 hN1 hS1).prob false false = (1/2 : ℝ) := h1_ff
    unfold nextJoint
    rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
    rw [e_ft, e_ff]
    unfold K2
    simp
  have h2_ff : nextJoint (nextJointLaw p K1 hN1 hS1) K2 false false = (1/2 : ℝ) := by
    have e_ft : (nextJointLaw p K1 hN1 hS1).prob false true = 0 := h1_ft
    have e_ff : (nextJointLaw p K1 hN1 hS1).prob false false = (1/2 : ℝ) := h1_ff
    unfold nextJoint
    rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
    rw [e_ft, e_ff]
    unfold K2
    simp
  have hm2_true : nextMarginalTheta (nextJointLaw p K1 hN1 hS1) K2 true = (1/2 : ℝ) := by
    unfold nextMarginalTheta
    rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
    rw [h2_tf, h2_tt]
    norm_num
  have hm2_false : nextMarginalTheta (nextJointLaw p K1 hN1 hS1) K2 false = (1/2 : ℝ) := by
    unfold nextMarginalTheta
    rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
    rw [h2_ff, h2_ft]
    norm_num
  have h_sum_next_true : (∑ t2 : Bool, nextJoint (nextJointLaw p K1 hN1 hS1) K2 t2 true) = (1/2 : ℝ) := by
    rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
    rw [h2_ft, h2_tt]
    norm_num
  have h_sum_marg : (∑ t3 : Bool, nextMarginalTheta (nextJointLaw p K1 hN1 hS1) K2 t3) = (1 : ℝ) := by
    rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
    rw [hm2_false, hm2_true]
    norm_num
  have h_rhs : nextMarginalTheta (nextJointLaw p K1 hN1 hS1) K2 true *
      (∑ t2 : Bool, nextJoint (nextJointLaw p K1 hN1 hS1) K2 t2 true /
        (∑ t3 : Bool, nextMarginalTheta (nextJointLaw p K1 hN1 hS1) K2 t3)) = (1/4 : ℝ) := by
    have h_sum_div : (∑ t2 : Bool, nextJoint (nextJointLaw p K1 hN1 hS1) K2 t2 true /
        (∑ t3 : Bool, nextMarginalTheta (nextJointLaw p K1 hN1 hS1) K2 t3)) =
        (∑ t2 : Bool, nextJoint (nextJointLaw p K1 hN1 hS1) K2 t2 true) /
          (∑ t3 : Bool, nextMarginalTheta (nextJointLaw p K1 hN1 hS1) K2 t3) := by
      rw [Finset.sum_div]
    rw [h_sum_div, hm2_true, h_sum_next_true, h_sum_marg]
    norm_num
  rw [h2_tt, h_rhs]
  norm_num

/-- The joint determines its marginals but not conversely: two distinct joints
share the same world-marginal. Witness on `Bool × Bool`: uniform `p1` vs
diagonal `p2`, both with world-marginal `1/2` on each `w`. -/
theorem joint_determines_marginals_but_not_converse :
    ∃ (p1 p2 : JointThetaWorld Bool Bool),
      (∀ w : Bool, (∑ t : Bool, p1.prob t w) = ∑ t : Bool, p2.prob t w) ∧
        p1.prob ≠ p2.prob := by
  let p1Prob : Bool → Bool → ℝ := fun _ _ => (1/4 : ℝ)
  let p2Prob : Bool → Bool → ℝ := fun t w => if t = w then (1/2 : ℝ) else (0 : ℝ)
  have h_univ : (Finset.univ : Finset Bool) = {false, true} := by decide
  have h1_nonneg : ∀ t w : Bool, 0 ≤ p1Prob t w := by
    intro t w
    unfold p1Prob
    norm_num
  have h1_sum : (∑ t : Bool, ∑ w : Bool, p1Prob t w) = 1 := by
    have h_inner : ∀ t : Bool, (∑ w : Bool, p1Prob t w) = (1/2 : ℝ) := by
      intro t
      rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
      unfold p1Prob
      norm_num
    have h_outer : (∑ t : Bool, (1/2 : ℝ)) = (1 : ℝ) := by
      rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]
      norm_num
    calc (∑ t : Bool, ∑ w : Bool, p1Prob t w) = ∑ t : Bool, (1/2 : ℝ) := by
          apply Finset.sum_congr rfl; intro t _; exact h_inner t
      _ = 1 := h_outer
  have h2_nonneg : ∀ t w : Bool, 0 ≤ p2Prob t w := by
    intro t w
    unfold p2Prob
    split <;> norm_num
  have h2_sum : (∑ t : Bool, ∑ w : Bool, p2Prob t w) = 1 := by
    have h_inner : ∀ t : Bool, (∑ w : Bool, p2Prob t w) = (1/2 : ℝ) := by
      intro t
      cases t
      · rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
        have f1 : p2Prob false false = (1/2 : ℝ) := rfl
        have f2 : p2Prob false true = (0 : ℝ) := rfl
        rw [f1, f2]
        norm_num
      · rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
        have f1 : p2Prob true false = (0 : ℝ) := rfl
        have f2 : p2Prob true true = (1/2 : ℝ) := rfl
        rw [f1, f2]
        norm_num
    have h_outer : (∑ t : Bool, (1/2 : ℝ)) = (1 : ℝ) := by
      rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]
      norm_num
    calc (∑ t : Bool, ∑ w : Bool, p2Prob t w) = ∑ t : Bool, (1/2 : ℝ) := by
          apply Finset.sum_congr rfl; intro t _; exact h_inner t
      _ = 1 := h_outer
  let p1 : JointThetaWorld Bool Bool := ⟨p1Prob, h1_nonneg, h1_sum⟩
  let p2 : JointThetaWorld Bool Bool := ⟨p2Prob, h2_nonneg, h2_sum⟩
  refine ⟨p1, p2, ?_, ?_⟩
  · intro w
    cases w
    · have hL : (∑ t : Bool, p1.prob t false) = (1/2 : ℝ) := by
        rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
        have e1 : p1.prob false false = (1/4 : ℝ) := rfl
        have e2 : p1.prob true false = (1/4 : ℝ) := rfl
        simp only [e1, e2]
        norm_num
      have hR : (∑ t : Bool, p2.prob t false) = (1/2 : ℝ) := by
        rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
        have f1 : p2.prob false false = (1/2 : ℝ) := rfl
        have f2 : p2.prob true false = (0 : ℝ) := rfl
        rw [f1, f2]
        norm_num
      rw [hL, hR]
    · have hL : (∑ t : Bool, p1.prob t true) = (1/2 : ℝ) := by
        rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
        have e1 : p1.prob false true = (1/4 : ℝ) := rfl
        have e2 : p1.prob true true = (1/4 : ℝ) := rfl
        simp only [e1, e2]
        norm_num
      have hR : (∑ t : Bool, p2.prob t true) = (1/2 : ℝ) := by
        rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
        have f1 : p2.prob false true = (0 : ℝ) := rfl
        have f2 : p2.prob true true = (1/2 : ℝ) := rfl
        rw [f1, f2]
        norm_num
      rw [hL, hR]
  · intro h
    have hc := congrFun (congrFun h true) true
    have e1 : p1.prob true true = (1/4 : ℝ) := rfl
    have e2 : p2.prob true true = (1/2 : ℝ) := rfl
    rw [e1, e2] at hc
    norm_num at hc

/-- Blueprint §15 `Q_set`: `Q = {q_j: divergence(q||q_nom)≤ρ, q=(1-ε)q_nom+ε r, r∈ support}`.
Finite version: policies are laws over a finite `A`, divergence is finite KL.
Nonempty via `r = q_nom`: then `q = q_nom` (`(1-ε)+ε = 1`) and
`D(q_nom‖q_nom) = 0 ≤ ρ` (each summand `q·log(q/q)` is `0`: `q=0` gives `0`, else
`log(q/q) = log 1 = 0`). -/
theorem Q_set_nonempty_when_nominal_feasible {A : Type} [Fintype A] [DecidableEq A]
    (q_nom : A → ℝ) (ρ ε : ℝ)
    (hq_nonneg : ∀ a, 0 ≤ q_nom a) (hq_sum : ∑ a : A, q_nom a = 1)
    (hρ : 0 ≤ ρ) :
    ∃ q : A → ℝ, (∃ r : A → ℝ, (∀ a, 0 ≤ r a) ∧ (∑ a : A, r a = 1) ∧
      q = fun a => (1 - ε) * q_nom a + ε * r a)
      ∧ (∑ a : A, q a * Real.log (q a / q_nom a)) ≤ ρ := by
  refine ⟨q_nom, ⟨q_nom, hq_nonneg, hq_sum, ?hmix⟩, ?hkl⟩
  · funext a
    ring
  · have hzero : (∑ a : A, q_nom a * Real.log (q_nom a / q_nom a)) = 0 := by
      apply Finset.sum_eq_zero
      intro a _
      by_cases hqa : q_nom a = 0
      · simp [hqa]
      · rw [div_self hqa, Real.log_one, mul_zero]
    rw [hzero]
    exact hρ

/-- Q-set radius calibration (Ou-Bi robust-MDP review Prop 5.2, via KuhnPost scout, ar5iv 2404.00940: L1 ambiguity `ρ_sa = √(2/n_sa · log(|S||A|2^|S|/δ))` with posterior-mean nominal `p̄_sa`; `ε` covers zero-probability-action rounding (Ganzfried Alg.3 epsilon) + KL-dual bisection tolerance Eq.18). `C` packs the log term `log(|S||A|2^|S|/δ)`; radius is nonneg and shrinks with `n` (more visits → tighter ball). Finite core: the closed-form rule + its monotonicity; the concentration inequality behind Prop 5.2 stays axiomatized. -/
noncomputable def robustRadius (n : ℕ) (C : ℝ) : ℝ :=
  Real.sqrt ((2 * C) / (n : ℝ))

theorem robustRadius_nonneg (n : ℕ) (C : ℝ) :
    0 ≤ robustRadius n C := by
  unfold robustRadius
  exact Real.sqrt_nonneg _

theorem robustRadius_antitone_n (n1 n2 : ℕ) (C : ℝ) (hC : 0 ≤ C)
    (hn1 : 0 < n1) (h_le : n1 ≤ n2) :
    robustRadius n2 C ≤ robustRadius n1 C := by
  unfold robustRadius
  apply Real.sqrt_le_sqrt
  have h1 : (0 : ℝ) ≤ 2 * C := by linarith
  have h2 : (n1 : ℝ) ≤ (n2 : ℝ) := Nat.cast_le.mpr h_le
  have hn1_pos : (0 : ℝ) < (n1 : ℝ) := Nat.cast_pos.mpr hn1
  exact div_le_div_of_nonneg_left h1 hn1_pos h2

/-- Hidden-hand marginalization: `p_next(x') = ∑_θ p_next(θ,x')` integrates out `θ`
for root's world belief, but the full joint is needed for next update. -/
theorem marginal_world_from_joint
    (p : JointThetaWorld Theta World) (K : World → World → Theta → ℝ) (x' : World) :
    (∑ t : Theta, nextJoint p K t x') = ∑ t : Theta, ∑ x : World, p.prob t x * K x x' t := by
  unfold nextJoint; rfl

end JointPosterior

end Hydra2.Blueprint.Opponent
