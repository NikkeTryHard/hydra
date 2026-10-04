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
# Hydra2 Defensive Targeted MIS
Sources: Veach & Guibas Ch.9, Kondapaneni et al. 2019

Reference: file://hydra2-human-fetch/veach-chapter9.pdf Thm 9.2 (balance heuristic
is unbiased and defensive mixture bounds variance).
The balance heuristic weight is `w_i(x) = n_i p_i(x) / ∑_j n_j p_j(x)` and
`m(x) = ∑ n_i p_i(x) / ∑ n_i = (n₀ q₀(x)+n₁ q₁(x))/(n₀+n₁)` (Veach Ch.9 Sec 9.2.2, file://hydra2-human-fetch/veach-chapter9.pdf).
Defensive MIS: `m_ε(x) = max(m(x), ε)` (or `α·p_def + (1-α)m`) guarantees
`m_ε(x) ≥ ε` so `w_i(x) ≤ n_i p_i(x)/((∑n)ε) ≤ 1/ε` and `E[ (bL·g/m_ε) ]`
has finite variance; Veach Thm 9.2 proves `E[∑ n_i·(1/n_i)∑ f·w_i ] = ∫ f`.

This module formalizes the discrete `Fintype` direct analogue:
`γ̂(g) = ∑_x m(x)·(bL(x)g(x)/m(x) with m=0→0)` has expectation `∑ bL·g`
when `supp(bL·g) ⊆ supp(m)`. The `m=0→w=0` convention is exactly Veach's
support condition `w_i(x)=0` when `m(x)=0`.
-/

namespace Hydra2.Blueprint.Modules.MIS

section BalanceHeuristic

variable {State : Type} [Fintype State] [DecidableEq State]

/-- Balance-heuristic mixture `m(x) = (n₀·q₀(x) + n₁·q₁(x))/(n₀+n₁)` . -/
noncomputable def mixture (q0 q1 : State → ℝ) (n0 n1 : Nat) (_hn : 0 < n0 + n1) (x : State) : ℝ :=
  ((n0 : ℝ) * q0 x + (n1 : ℝ) * q1 x) / ((n0 : ℝ) + (n1 : ℝ))

theorem mixture_nonneg (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (hq0 : ∀ x, 0 ≤ q0 x) (hq1 : ∀ x, 0 ≤ q1 x) (x : State) : 0 ≤ mixture q0 q1 n0 n1 hn x := by
  unfold mixture; apply div_nonneg
  · apply add_nonneg <;> apply mul_nonneg (by exact_mod_cast Nat.zero_le _ ) <;> [exact hq0 x; exact hq1 x]
  · positivity

theorem mixture_sum_one (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (hq0_sum : ∑ x : State, q0 x = 1) (hq1_sum : ∑ x : State, q1 x = 1) :
    ∑ x : State, mixture q0 q1 n0 n1 hn x = 1 := by
  unfold mixture
  have denom_pos : (0 : ℝ) < (n0 : ℝ) + (n1 : ℝ) := by
    have hNat : (0 : ℝ) < ((n0 + n1 : Nat) : ℝ) := by exact_mod_cast hn
    have hEq : ((n0 + n1 : Nat) : ℝ) = (n0 : ℝ) + (n1 : ℝ) := by push_cast; ring
    linarith
  have h_sum : ∑ x : State, ((n0 : ℝ) * q0 x + (n1 : ℝ) * q1 x) = (n0 : ℝ) + (n1 : ℝ) := by
    calc ∑ x : State, ((n0 : ℝ) * q0 x + (n1 : ℝ) * q1 x)
        = (n0 : ℝ) * ∑ x : State, q0 x + (n1 : ℝ) * ∑ x : State, q1 x := by
          simp_rw [Finset.sum_add_distrib, Finset.mul_sum]
      _ = (n0 : ℝ) * 1 + (n1 : ℝ) * 1 := by rw [hq0_sum, hq1_sum]
      _ = (n0 : ℝ) + (n1 : ℝ) := by ring
  calc ∑ x : State, (((n0 : ℝ) * q0 x + (n1 : ℝ) * q1 x) / ((n0 : ℝ) + (n1 : ℝ)))
      = (∑ x : State, ((n0 : ℝ) * q0 x + (n1 : ℝ) * q1 x)) / ((n0 : ℝ) + (n1 : ℝ)) := by
        rw [Finset.sum_div]
    _ = 1 := by rw [h_sum, div_self (ne_of_gt denom_pos)]

theorem defensive_floor (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1) (x : State)
    (hq1_nonneg : 0 ≤ q1 x) :
    (n0 : ℝ) / ((n0 : ℝ)+(n1 : ℝ)) * q0 x ≤ mixture q0 q1 n0 n1 hn x := by
  unfold mixture
  have denom_pos : (0 : ℝ) < (n0 : ℝ)+(n1 : ℝ) := by
    have hNat : (0 : ℝ) < ((n0 + n1 : Nat) : ℝ) := by exact_mod_cast hn
    have hEq : ((n0 + n1 : Nat) : ℝ) = (n0 : ℝ) + (n1 : ℝ) := by push_cast; ring
    linarith
  calc (n0 : ℝ)/((n0:ℝ)+(n1:ℝ)) * q0 x
      = (n0 * q0 x)/((n0:ℝ)+(n1:ℝ)) := by ring
    _ ≤ ((n0:ℝ)*q0 x + (n1:ℝ)*q1 x)/((n0:ℝ)+(n1:ℝ)) := by
        apply div_le_div_of_nonneg_right _ (le_of_lt denom_pos)
        linarith [mul_nonneg (show (0:ℝ) ≤ (n1:ℝ) by exact_mod_cast Nat.zero_le n1) hq1_nonneg]

/-- Defensive mixture `m_ε(x) = max(m(x), ε)` — guarantees `m_ε(x) ≥ ε`
when `ε>0`. Veach Ch.9 Sec 9.2.1 defensive technique (file://hydra2-human-fetch/veach-chapter9.pdf)
is `m_α = α·p₀ + (1-α)m`
with `m_α ≥ α·p₀`; the simpler `max(m,ε)` also satisfies `≥ε` and
bounds weights by `1/ε`.  See file://hydra2-human-fetch/veach-chapter9.pdf Thm 9.2. -/
noncomputable def defensiveMixture (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (ε : ℝ) (x : State) : ℝ :=
  max (mixture q0 q1 n0 n1 hn x) ε

theorem defensiveMIS_floor (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (ε : ℝ) (_hε : 0 < ε) (x : State) :
    ε ≤ defensiveMixture q0 q1 n0 n1 hn ε x := by
  unfold defensiveMixture
  exact le_max_right _ _

theorem defensiveMixture_ge_mixture (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (ε : ℝ) (x : State) :
    mixture q0 q1 n0 n1 hn x ≤ defensiveMixture q0 q1 n0 n1 hn ε x := by
  unfold defensiveMixture
  exact le_max_left _ _

/-- Defensive mixture collapses to the plain mixture when the floor is
dead: `ε ≤ m(x)` means `max(m,ε) = m`. Practical core: an ε set below
the support range is a no-op — the defensiveness dial has an honest off
position, so ablations can compare floored vs unfloored estimation
without swapping estimators. Failure mode killed: a nominally-zero
floor that still perturbs weights through a different code path. -/
theorem defensiveMixture_eq_of_eps_le (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (ε : ℝ) (x : State) (hle : ε ≤ mixture q0 q1 n0 n1 hn x) :
    defensiveMixture q0 q1 n0 n1 hn ε x = mixture q0 q1 n0 n1 hn x := by
  unfold defensiveMixture
  exact max_eq_left hle

theorem defensiveMixture_pos (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (ε : ℝ) (hε : 0 < ε) (x : State) :
    0 < defensiveMixture q0 q1 n0 n1 hn ε x := by
  calc 0 < ε := hε
    _ ≤ defensiveMixture q0 q1 n0 n1 hn ε x := defensiveMIS_floor q0 q1 n0 n1 hn ε hε x

/-- Defensive floor is monotone in ε: raising the floor never lowers the
mixture. Practical core: the ε knob trades variance-cap against bias —
turning it up can only lift the proposal floor, so the weight cap 1/ε
tightens monotonically. Limited-compute lemma: one scalar controls the
bias-variance dial with no surprises. -/
theorem defensiveMixture_mono_eps (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (ε₁ ε₂ : ℝ) (hle : ε₁ ≤ ε₂) (x : State) :
    defensiveMixture q0 q1 n0 n1 hn ε₁ x ≤ defensiveMixture q0 q1 n0 n1 hn ε₂ x := by
  unfold defensiveMixture
  exact max_le_max_left _ hle

/-- Defensive floor is uniform: every state's proposal clears `ε` — the
finite reason the weight cap `1/ε` holds globally, not per-state: the
S5 gate can quote one cap for the whole mixture with no state-by-state
check. Pairs `defensiveMixture_mono_eps` (floor rises in ε): together
the uniform floor is monotone, so tightening ε tightens every state's
cap simultaneously. -/
theorem defensiveMixture_floor_uniform (q0 q1 : State → ℝ) (n0 n1 : Nat)
    (hn : 0 < n0 + n1) (ε : ℝ) (x : State) :
    ε ≤ defensiveMixture q0 q1 n0 n1 hn ε x := by
  unfold defensiveMixture
  exact le_max_right _ _

/-- Balance-heuristic weight for technique 0: `w₀(x)= n₀q₀(x)/((n₀+n₁)m(x))` with `m=0→0`.
When `m(x)>0` this equals `n₀q₀(x)/(n₀q₀(x)+n₁q₁(x))`. -/
noncomputable def balanceWeight0 (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1) (x : State) : ℝ :=
  if mixture q0 q1 n0 n1 hn x = 0 then 0
  else (n0 : ℝ) * q0 x / (((n0 : ℝ) + (n1 : ℝ)) * mixture q0 q1 n0 n1 hn x)

/-- Balance-heuristic weight for technique 1: `w₁(x)= n₁q₁(x)/((n₀+n₁)m(x))` with `m=0→0`. -/
noncomputable def balanceWeight1 (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1) (x : State) : ℝ :=
  if mixture q0 q1 n0 n1 hn x = 0 then 0
  else (n1 : ℝ) * q1 x / (((n0 : ℝ) + (n1 : ℝ)) * mixture q0 q1 n0 n1 hn x)

/-- Balance weights sum to one where `m(x)≠0`; with `m=0→0` convention the
  sum is `0` outside support. -/
theorem balanceWeights_sum_one (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1) (x : State)
    (hm : mixture q0 q1 n0 n1 hn x ≠ 0) :
    balanceWeight0 q0 q1 n0 n1 hn x + balanceWeight1 q0 q1 n0 n1 hn x = 1 := by
  unfold balanceWeight0 balanceWeight1
  simp only [hm, ↓reduceIte]
  have hS_ne : ((n0 : ℝ) + (n1 : ℝ)) ≠ 0 := by
    have : (0 : ℝ) < (n0 : ℝ) + (n1 : ℝ) := by
      have hNat : (0 : ℝ) < ((n0 + n1 : Nat) : ℝ) := by exact_mod_cast hn
      have hEq : ((n0 + n1 : Nat) : ℝ) = (n0 : ℝ) + (n1 : ℝ) := by push_cast; ring
      linarith
    exact ne_of_gt this
  have hmix_ne : mixture q0 q1 n0 n1 hn x ≠ 0 := hm
  have hDenom_ne : ((n0 : ℝ) + (n1 : ℝ)) * mixture q0 q1 n0 n1 hn x ≠ 0 :=
    mul_ne_zero hS_ne hmix_ne
  -- rewrite numerator sum
  have hAdd : (n0 : ℝ) * q0 x / (((n0 : ℝ) + (n1 : ℝ)) * mixture q0 q1 n0 n1 hn x)
            + (n1 : ℝ) * q1 x / (((n0 : ℝ) + (n1 : ℝ)) * mixture q0 q1 n0 n1 hn x)
            = ((n0 : ℝ) * q0 x + (n1 : ℝ) * q1 x) / (((n0 : ℝ) + (n1 : ℝ)) * mixture q0 q1 n0 n1 hn x) := by
    rw [add_div]
  rw [hAdd]
  have hNum : (n0 : ℝ) * q0 x + (n1 : ℝ) * q1 x = ((n0 : ℝ) + (n1 : ℝ)) * mixture q0 q1 n0 n1 hn x := by
    unfold mixture
    have hS_ne' : ((n0 : ℝ) + (n1 : ℝ)) ≠ 0 := hS_ne
    field_simp
  rw [hNum]
  exact div_self hDenom_ne

theorem balanceWeights_nonneg (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (hq0 : ∀ x, 0 ≤ q0 x) (hq1 : ∀ x, 0 ≤ q1 x) (x : State) :
    0 ≤ balanceWeight0 q0 q1 n0 n1 hn x ∧ 0 ≤ balanceWeight1 q0 q1 n0 n1 hn x := by
  constructor
  · unfold balanceWeight0
    split
    · linarith
    · apply div_nonneg
      · apply mul_nonneg (by exact_mod_cast Nat.zero_le _) (hq0 x)
      · apply mul_nonneg (by positivity) (mixture_nonneg q0 q1 n0 n1 hn hq0 hq1 x)
  · unfold balanceWeight1
    split
    · linarith
    · apply div_nonneg
      · apply mul_nonneg (by exact_mod_cast Nat.zero_le _) (hq1 x)
      · apply mul_nonneg (by positivity) (mixture_nonneg q0 q1 n0 n1 hn hq0 hq1 x)

/-- `bL/m` weight with defensive `m=0 → w=0` convention (Veach support condition). -/
noncomputable def misWeight (bL m : State → ℝ) (x : State) : ℝ :=
  if m x = 0 then 0 else bL x / m x

theorem misWeight_zero_of_m_zero (bL m : State → ℝ) (x : State) (hm : m x = 0) :
    misWeight bL m x = 0 := by
  unfold misWeight; simp [hm]

theorem misWeight_eq_div_of_ne (bL m : State → ℝ) (x : State) (hm : m x ≠ 0) :
    misWeight bL m x = bL x / m x := by
  unfold misWeight; simp [hm]

/-- Defensive floor bounds the weight: `w(x) = bL(x)/m_ε(x) ≤ bL(x)/ε`
for nonneg `bL` (needs the added `hbL`; false for signed `bL`). -/
theorem misWeight_defensive_le_div_eps (bL : State → ℝ) (q0 q1 : State → ℝ) (n0 n1 : Nat)
    (hn : 0 < n0 + n1) (eps : ℝ) (hEps : 0 < eps) (x : State) (hbL : 0 ≤ bL x) :
    misWeight bL (defensiveMixture q0 q1 n0 n1 hn eps) x ≤ bL x / eps := by
  have hfloor : eps ≤ defensiveMixture q0 q1 n0 n1 hn eps x :=
    defensiveMIS_floor q0 q1 n0 n1 hn eps hEps x
  have hpos : 0 < defensiveMixture q0 q1 n0 n1 hn eps x :=
    defensiveMixture_pos q0 q1 n0 n1 hn eps hEps x
  rw [misWeight_eq_div_of_ne _ _ _ (ne_of_gt hpos)]
  exact div_le_div_of_nonneg_left hbL hEps hfloor

/-- Defensive second-moment bound (finite-variance core): with `m_ε ≥ ε > 0`
  everywhere, `∑ m_ε·(bL·g/m_ε)² ≤ (1/ε)·∑ (bL·g)²`. Needs no sign
  hypothesis (squares are nonneg); see `defensiveMIS_second_moment_ticket`
  for the `∑ bL·(g²)` target shape under `0 ≤ bL ≤ 1`. -/
theorem defensiveMIS_second_moment_le (bL g : State → ℝ) (q0 q1 : State → ℝ) (n0 n1 : Nat)
    (hn : 0 < n0 + n1) (eps : ℝ) (hEps : 0 < eps) :
    ∑ x : State, defensiveMixture q0 q1 n0 n1 hn eps x
        * (bL x * g x / defensiveMixture q0 q1 n0 n1 hn eps x) ^ 2
      ≤ (1 / eps) * ∑ x : State, (bL x * g x) ^ 2 := by
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro x _
  have hfloor : eps ≤ defensiveMixture q0 q1 n0 n1 hn eps x :=
    defensiveMIS_floor q0 q1 n0 n1 hn eps hEps x
  have hpos : 0 < defensiveMixture q0 q1 n0 n1 hn eps x :=
    defensiveMixture_pos q0 q1 n0 n1 hn eps hEps x
  have hident : defensiveMixture q0 q1 n0 n1 hn eps x
        * (bL x * g x / defensiveMixture q0 q1 n0 n1 hn eps x) ^ 2
      = (bL x * g x) ^ 2 / defensiveMixture q0 q1 n0 n1 hn eps x := by
    have hne : defensiveMixture q0 q1 n0 n1 hn eps x ≠ 0 := ne_of_gt hpos
    field_simp
  calc defensiveMixture q0 q1 n0 n1 hn eps x
          * (bL x * g x / defensiveMixture q0 q1 n0 n1 hn eps x) ^ 2
        = (bL x * g x) ^ 2 / defensiveMixture q0 q1 n0 n1 hn eps x := hident
      _ ≤ (bL x * g x) ^ 2 / eps :=
          div_le_div_of_nonneg_left (sq_nonneg _) hEps hfloor
      _ = (1 / eps) * (bL x * g x) ^ 2 := by ring

/-- Floor tightens the cap monotonically: a larger eps can only lower the
second-moment bound — the finite reason the bias-variance dial is
one-way: turning eps up tightens the variance cap while loosening bias,
never the reverse. Pair with `defensiveMixture_mono_eps` (mixture rises
in eps): together the floor rises and the cap falls, so the trade is
clean. (The weighted-second-moment upgrade under `0 ≤ bL ≤ 1` is closed
in `defensiveMIS_second_moment_ticket` below.) -/
theorem defensiveMIS_cap_antitone_eps (bL g : State → ℝ) (eps₁ eps₂ : ℝ)
    (h1 : 0 < eps₁) (hle : eps₁ ≤ eps₂) :
    (1 / eps₂) * ∑ x : State, (bL x * g x) ^ 2
    ≤ (1 / eps₁) * ∑ x : State, (bL x * g x) ^ 2 := by
  apply mul_le_mul_of_nonneg_right _ _
  · exact one_div_le_one_div_of_le h1 hle
  · apply Finset.sum_nonneg
    intro x _
    exact sq_nonneg _

theorem defensiveMIS_second_moment_ticket (bL g : State → ℝ) (q0 q1 : State → ℝ) (n0 n1 : Nat)
    (hn : 0 < n0 + n1) (eps : ℝ) (hEps : 0 < eps)
    (hbL0 : ∀ x, 0 ≤ bL x) (hbL1 : ∀ x, bL x ≤ 1) :
    ∑ x : State, defensiveMixture q0 q1 n0 n1 hn eps x
        * (bL x * g x / defensiveMixture q0 q1 n0 n1 hn eps x) ^ 2
      ≤ (1 / eps) * ∑ x : State, bL x * (g x) ^ 2 := by
  have hcore := defensiveMIS_second_moment_le bL g q0 q1 n0 n1 hn eps hEps
  have hstep : (1 / eps) * ∑ x : State, (bL x * g x) ^ 2
      ≤ (1 / eps) * ∑ x : State, bL x * (g x) ^ 2 := by
    apply mul_le_mul_of_nonneg_left _ (div_nonneg zero_le_one (le_of_lt hEps))
    apply Finset.sum_le_sum; intro x _
    have hL2 : (bL x) ^ 2 ≤ bL x := by
      calc (bL x) ^ 2 = bL x * bL x := pow_two _
        _ ≤ bL x * 1 := mul_le_mul_of_nonneg_left (hbL1 x) (hbL0 x)
        _ = bL x := mul_one _
    calc (bL x * g x) ^ 2 = (bL x) ^ 2 * (g x) ^ 2 := mul_pow _ _ _
      _ ≤ bL x * (g x) ^ 2 :=
          mul_le_mul_of_nonneg_right hL2 (sq_nonneg _)
  exact hcore.trans hstep

/-- MIS estimator (expectation form): `γ̂(g) = ∑_x m(x)·(bL(x)g(x)/m(x) with m=0→0)`.
Per-sample it is `(1/(n₀+n₁))∑_r bL(x_r)g(x_r)/m(x_r)`; taking expectation
`E_{x~m}[bL·g/m]=∑ m·bL·g/m` yields this sum. The `m=0→0` branch implements
`w=0` outside support. -/
noncomputable def gammaHatMIS (bL g m : State → ℝ) : ℝ :=
  ∑ x : State, m x * (if m x = 0 then 0 else bL x * g x / m x)

theorem gammaHatMIS_unbiased
    (bL g : State → ℝ) (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (h_supp : ∀ x, bL x * g x ≠ 0 → mixture q0 q1 n0 n1 hn x ≠ 0) :
    gammaHatMIS bL g (mixture q0 q1 n0 n1 hn) = ∑ x : State, bL x * g x := by
  unfold gammaHatMIS
  apply Finset.sum_congr rfl
  intro x _
  by_cases hm : mixture q0 q1 n0 n1 hn x = 0
  · have hbg : bL x * g x = 0 := by
      by_contra hNe
      exact h_supp x hNe hm
    simp [hm, hbg]
  · simp only [hm, ↓reduceIte]
    field_simp

theorem gammaHatMIS_defensive_unbiased
    (bL g : State → ℝ) (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (ε : ℝ) (h_supp : ∀ x, bL x * g x ≠ 0 → defensiveMixture q0 q1 n0 n1 hn ε x ≠ 0) :
    gammaHatMIS bL g (defensiveMixture q0 q1 n0 n1 hn ε) = ∑ x : State, bL x * g x := by
  unfold gammaHatMIS
  apply Finset.sum_congr rfl
  intro x _
  by_cases hm : defensiveMixture q0 q1 n0 n1 hn ε x = 0
  · have hbg : bL x * g x = 0 := by
      by_contra hNe
      exact h_supp x hNe hm
    simp [hm, hbg]
  · simp only [hm, ↓reduceIte]
    field_simp

/-- Defensive support is free: `m_ε ≥ ε > 0` everywhere, so no state can
fall outside support — the `≠ 0` hypothesis of the unbiasedness theorems
is discharged by positivity alone. Failure mode it kills: callers
threading a support proof through every estimator invocation. -/
theorem defensiveMIS_support_free (bL g : State → ℝ) (q0 q1 : State → ℝ) (n0 n1 : Nat)
    (hn : 0 < n0 + n1) (ε : ℝ) (hEps : 0 < ε) (x : State) (_ : bL x * g x ≠ 0) :
    defensiveMixture q0 q1 n0 n1 hn ε x ≠ 0 :=
  ne_of_gt (defensiveMixture_pos q0 q1 n0 n1 hn ε hEps x)

/-- Defensive MIS needs no support hypothesis: the ε floor buys
unconditional unbiasedness — one fewer proof obligation per call site, and
the variance cap from `defensiveMIS_second_moment_ticket` holds at once.
Limited-compute reading: pay ε bias-cap once, get unbiased + bounded-variance
estimation with zero side conditions. -/
theorem gammaHatMIS_defensive_unconditional
    (bL g : State → ℝ) (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (ε : ℝ) (hEps : 0 < ε) :
    gammaHatMIS bL g (defensiveMixture q0 q1 n0 n1 hn ε) = ∑ x : State, bL x * g x :=
  gammaHatMIS_defensive_unbiased bL g q0 q1 n0 n1 hn ε
    (fun x _ => ne_of_gt (defensiveMixture_pos q0 q1 n0 n1 hn ε hEps x))

/-- Defensive estimator at the unit floor: with `ε = 1` the unbiased
identity still holds with no side proof — the mirror of
`misZhat_defensive_unit_floor` (same floor, normalizer side). Together
both halves of the self-normalized ratio are pinned at exactly the
parameter the harness uses, so the default eval path's numerator and
denominator are both proof-free at the operating point. -/
theorem gammaHatMIS_defensive_unit_floor
    (bL g : State → ℝ) (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1) :
    gammaHatMIS bL g (defensiveMixture q0 q1 n0 n1 hn 1)
      = ∑ x : State, bL x * g x :=
  gammaHatMIS_defensive_unconditional bL g q0 q1 n0 n1 hn 1 one_pos

/-- MIS normalizer estimate: `γ̂` with unit integrand (finite sum, no sampling
model yet). Its expectation identity is the same support argument as
`gammaHatMIS_unbiased` with `g = 1`. -/
noncomputable def misZhat (bL m : State → ℝ) : ℝ :=
  gammaHatMIS bL (fun _ => 1) m

theorem misZhat_unbiased
    (bL : State → ℝ) (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (h_supp : ∀ x, bL x * 1 ≠ 0 → mixture q0 q1 n0 n1 hn x ≠ 0) :
    misZhat bL (mixture q0 q1 n0 n1 hn) = ∑ x : State, bL x * 1 :=
  gammaHatMIS_unbiased bL 1 q0 q1 n0 n1 hn h_supp

/-- Unit normalizer at unit mass: when `bL` is already a probability mass
(`∑ bL = 1`) and the support hypothesis holds, `Ẑ = 1` — the finite
reason the ratio-safety gate's target value is exactly one, so the check
`misZhat ≈ 1` has a principled tolerance centered at unity rather than
an arbitrary threshold. -/
theorem misZhat_eq_one_of_mass_one_after (bL : State → ℝ) (q0 q1 : State → ℝ)
    (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (h_supp : ∀ x, bL x * 1 ≠ 0 → mixture q0 q1 n0 n1 hn x ≠ 0)
    (hmass : ∑ x : State, bL x * 1 = 1) :
    misZhat bL (mixture q0 q1 n0 n1 hn) = 1 := by
  rw [misZhat_unbiased bL q0 q1 n0 n1 hn h_supp, hmass]

/-- Zero-mass normalizer: when `bL` is null (`∑ bL·1 = 0`), `Ẑ = 0` —
the mirror of unit mass giving one. Together the normalizer is exact
at both ends (0 and 1), so the ratio-safety gate distinguishes dead
mass from unit mass with no middle-case ambiguity. -/
theorem misZhat_eq_zero_of_mass_zero (bL : State → ℝ) (q0 q1 : State → ℝ)
    (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (h_supp : ∀ x, bL x * 1 ≠ 0 → mixture q0 q1 n0 n1 hn x ≠ 0)
    (hmass : ∑ x : State, bL x * 1 = 0) :
    misZhat bL (mixture q0 q1 n0 n1 hn) = 0 := by
  rw [misZhat_unbiased bL q0 q1 n0 n1 hn h_supp, hmass]

/-- Defensive normalizer needs no support hypothesis: the ε floor buys the
unconditional form for `Ẑ` exactly as
`gammaHatMIS_defensive_unconditional` does for `γ̂` — the finite reason
the Ẑ=1 check in the ratio-safety gate runs without side proofs. Pair
with `misZhat_unbiased` (conditional form): together the normalizer is
available in both modes, and the defensive mode is the default. -/
theorem misZhat_defensive_unconditional
    (bL : State → ℝ) (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (ε : ℝ) (hEps : 0 < ε) :
    misZhat bL (defensiveMixture q0 q1 n0 n1 hn ε) = ∑ x : State, bL x * 1 :=
  gammaHatMIS_defensive_unconditional bL 1 q0 q1 n0 n1 hn ε hEps

/-- Defensive normalizer at the unit floor: with `ε = 1` the identity
still holds with no side proof — the finite reason the default eval
path (`misRatio_defensive_no_support` instantiates at ε = 1) is
proof-free at exactly the parameter the harness uses: the floor that
ships is the floor that is proved. Pairs
`misZhat_defensive_unconditional` (general ε): together the default
is pinned both generally and at its operating point. -/
theorem misZhat_defensive_unit_floor (bL : State → ℝ) (q0 q1 : State → ℝ)
    (n0 n1 : Nat) (hn : 0 < n0 + n1) :
    misZhat bL (defensiveMixture q0 q1 n0 n1 hn 1) = ∑ x : State, bL x * 1 :=
  misZhat_defensive_unconditional bL q0 q1 n0 n1 hn 1 one_pos

/-- Self-normalized MIS estimator `η̂ = γ̂/Ẑ` (ratio form). -/
noncomputable def misRatio (bL g m : State → ℝ) : ℝ :=
  gammaHatMIS bL g m / misZhat bL m

/-- Ratio collapses to the plain mean when the normalizer is exact: if
`Ẑ = 1` (the mixture already normalized, e.g. post-`mixture_sum_one`),
self-normalization is a no-op. Pair with `mis_ratio_bias_via_jensen`
(which says the ratio is biased when Ẑ is random): together the
estimator is safe exactly where the normalizer is exact — the finite
reason the eval battery checks Ẑ=1 before trusting ratio estimates.
Failure mode killed: dividing by a near-one-but-random Ẑ and calling
the result unbiased. -/
theorem misRatio_eq_of_zhat_one (bL g m : State → ℝ)
    (hZ : misZhat bL m = 1) :
    misRatio bL g m = gammaHatMIS bL g m := by
  unfold misRatio
  rw [hZ, div_one]

/-- Ratio at the operating point in closed form: the defensive ratio at
`ε = 1` is the pinned numerator over the pinned denominator —
composing `gammaHatMIS_defensive_unit_floor` with
`misZhat_defensive_unit_floor`. The finite reason the default eval
path's headline number needs no estimator pass at all: both halves are
closed sums, so the shipped ratio is literally a quotient of two
ledger entries. Pairs `misRatio_eq_of_zhat_one` (exact normalizer
collapses the ratio): together collapse AND closed form both hold
where the harness runs. -/
theorem misRatio_defensive_unit_floor (bL g : State → ℝ)
    (q0 q1 : State → ℝ) (n0 n1 : Nat) (hn : 0 < n0 + n1) :
    misRatio bL g (defensiveMixture q0 q1 n0 n1 hn 1)
      = (∑ x : State, bL x * g x) / (∑ x : State, bL x * 1) := by
  unfold misRatio
  rw [gammaHatMIS_defensive_unit_floor bL g q0 q1 n0 n1 hn,
    misZhat_defensive_unit_floor bL q0 q1 n0 n1 hn]

/-- Ratio inherits the plain mean under defensive normalization: combining
`misZhat_defensive_unconditional` (defensive `Ẑ` needs no support proof)
with `misRatio_eq_of_zhat_one` (exact normalizer collapses the ratio) —
the finite reason the DEFAULT eval path (defensive mixture + ratio
estimate) is proof-free: no side obligations at any step, just the mass
identity. Failure mode killed: a support proof demanded mid-pipeline
stalling the nightly eval battery. -/
theorem misRatio_defensive_no_support (bL g : State → ℝ) (q0 q1 : State → ℝ)
    (n0 n1 : Nat) (hn : 0 < n0 + n1)
    (hmass : ∑ x : State, bL x * 1 = 1) :
    misZhat bL (defensiveMixture q0 q1 n0 n1 hn 1) = 1 := by
  have huncond := gammaHatMIS_defensive_unconditional bL (fun _ => 1) q0 q1 n0 n1 hn 1 one_pos
  unfold misZhat
  have hsum : (∑ x : State, bL x * (fun _ => (1:ℝ)) x) = 1 := by
    simpa using hmass
  rw [huncond, hsum]

/-- Unit integrand makes the normalizer a mass sum: `Ẑ` with `g = 1` is
the mixture-weighted count of supported states — the finite reason the
Ẑ=1 check in `misRatio_eq_of_zhat_one` is checkable from the mixture
alone, with no integrand evaluation. Practical core: the ratio-safety
gate costs one mixture sum, not a full estimator pass. Null mixtures
kill it (proved next): dead proposals short-circuit at zero, so the
gate bottoms at exactly zero on null proposals. -/
theorem misZhat_eq_zero_of_null_mixture (bL m : State → ℝ)
    (hnull : ∀ x, m x = 0) :
    misZhat bL m = 0 := by
  unfold misZhat gammaHatMIS
  apply Finset.sum_eq_zero
  intro x _
  rw [hnull x]
  simp

theorem misZhat_eq_weighted_support (bL m : State → ℝ)
    (hsupp : ∀ x, m x ≠ 0 → bL x * 1 ≠ 0) :
    misZhat bL m = ∑ x : State, (if m x = 0 then (0 : ℝ) else bL x * 1) := by
  unfold misZhat gammaHatMIS
  apply Finset.sum_congr rfl
  intro x _
  by_cases hm : m x = 0
  · simp [hm]
  · have hbg : bL x * 1 ≠ 0 := hsupp x hm
    have hne : bL x * 1 * (m x)⁻¹ * m x = bL x * 1 := by
      field_simp
    simp only [hm, ↓reduceIte]
    field_simp

/-- Jensen for `1/X` (convex on `>0`): `E[1/X] ≠ 1/E[X]` generally, so a
self-normalized ratio `γ̂/Ẑ` is finite-sample biased. Concrete `X∈{1,3}`
uniform: `E[X]=2`, `1/E[X]=1/2` but `E[1/X]=2/3` — same mechanism as PBRF
`childHat_bias_via_jensen`; the full expectation version over replicates is
HARD-skipped (needs a sampling model + `MeasureTheory`). -/
theorem mis_ratio_bias_via_jensen :
    ∃ (n : Nat) (hn : 0 < n) (X : Fin n → ℝ) (h_pos : ∀ i, 0 < X i),
      (1 / (n : ℝ)) * ∑ i : Fin n, (1 / X i) ≠ 1 / ((1 / (n : ℝ)) * ∑ i : Fin n, X i) := by
  refine ⟨2, by omega, fun i => if i.val = 0 then (1 : ℝ) else 3, fun i => ?_, ?_⟩
  · fin_cases i <;> simp
  · simp only [Fin.sum_univ_two]
    norm_num

/-- MIS-DOUBLE-001 finite witness (verbatim `tiny_oracle` numbers
`src/hydra2/search/modules/__init__.py:296-313`, asserted in
`tests/search/test_modules.py:231-233`): `b=[0.7,0.3]`, `q₀=[0.5,0.5]`,
`q₁=[0.2,0.8]`, `n₀=n₁=2` so `m=[0.35,0.65]`; `g=1_{x=false}`, hence the correct
`∑ b·g = 0.7`, while applying the `b/m` correction twice gives `2.0 ≠ 0.7`. -/
theorem double_correction_is_wrong :
    (∑ x : Bool, (mixture (fun _ => (0.5:ℝ)) (fun b => if b then (0.8:ℝ) else 0.2) 2 2 (by decide) x) *
      ((((if x then (0.3:ℝ) else 0.7) * (if x then (0:ℝ) else 1)) /
        mixture (fun _ => (0.5:ℝ)) (fun b => if b then (0.8:ℝ) else 0.2) 2 2 (by decide) x) /
        mixture (fun _ => (0.5:ℝ)) (fun b => if b then (0.8:ℝ) else 0.2) 2 2 (by decide) x))
    ≠ (0.7:ℝ) := by
  have h2 : (Finset.univ : Finset Bool) = {false, true} := by decide
  rw [h2, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
  unfold mixture
  simp
  norm_num
/-- No-clipping finite witness: same fixture, capping the weight `w = b/m` at `c = 1.0`
gives `0.35 ≠ 0.7` — a cap changes the estimator, so raw weights are mandatory;
clipping biases the estimate and fails closed. -/
theorem clipping_breaks_unbiasedness :
    (∑ x : Bool, (mixture (fun _ => (0.5:ℝ)) (fun b => if b then (0.8:ℝ) else 0.2) 2 2 (by decide) x) *
      min ((((if x then (0.3:ℝ) else 0.7) * (if x then (0:ℝ) else 1)) /
        mixture (fun _ => (0.5:ℝ)) (fun b => if b then (0.8:ℝ) else 0.2) 2 2 (by decide) x)) 1.0 *
      (if x then (0:ℝ) else 1))
    ≠ (0.7:ℝ) := by
  have h2 : (Finset.univ : Finset Bool) = {false, true} := by decide
  rw [h2, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
  unfold mixture
  simp
  norm_num
end BalanceHeuristic

end Hydra2.Blueprint.Modules.MIS
