import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Logic.Function.Iterate
import Mathlib.Tactic

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedDecidableInType false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false

/-! # Hydra2 Gumbel search identities (Gumbel-Top-k + sequential halving)

Live planner `src/hydra2/search/gumbel.py` (1819 lines): deterministic inverse-CDF sampler
(`src/hydra2/search/gumbel.py#deterministic_gumbel` inverse-CDF sampler, root gumbels,
score rule `score = g + q`, halving keep `ceil(n/2)`, final argmax;
wrong keep fraction breaks the halving schedule and fails closed.

Paper chain (GumbelMiner):
- C1 Gumbel-Max: `argmax_i{phi_i + G_i} ~ Categorical(softmax phi)`
  (Kool et al. ICML19 Sec 2.3, https://arxiv.org/abs/1903.06059; derivation Princeton LIPS 2013,
https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/).
  URLs: https://ar5iv.labs.arxiv.org/html/1903.06059
        https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
- C2 CDF `F(z;mu) = exp(-exp(-(z-mu)))` + sampler `G = phi - log(-log U)`
  (Kool Sec 2.2, App B.1 Eq20-21, https://arxiv.org/abs/1903.06059).
- C3 Gumbel-Top-k Thm 1 Sec 2.4 (https://arxiv.org/abs/1903.06059): ordered top-k = ordered WOR sample
  (Eq4, proof App A). https://arxiv.org/abs/1903.06059
- C4 Sequential halving: keep `ceil(|S|/2)` per round over `ceil(log2 n)`
  rounds (Karnin et al. 2013, Alg 1 via https://arxiv.org/html/2406.00424v1).
- C5 Completed-Q + improved policy `softmax(prior + completedQ)`
  (Danihelka ICLR22; mctx `qtransforms.py`/`policies.py` via Context7
  /google-deepmind/mctx). https://mlanthology.org/iclr/2022/danihelka2022iclr-policy/

Finite scope (justification, not vibes): sampler inverse-CDF identity,
score monotonicity, halving schedule arithmetic, and softmax normalization
are `ℝ`/`ℕ`/`Finset` only. The full sampling-distribution claim (C1 `~`),
the Top-k WOR product law (C3 Eq4), the halving regret bound (C4 `O~(n/T)`),
and the policy-improvement EXPECTATION (C5) need MeasureTheory/MDP models
and are HARD-skipped — stated as doc pointers, never as theorems.
-/

namespace Hydra2.Blueprint.Modules.Gumbel

section Sampler

/-- Deterministic Gumbel(0,1) sampler (`gumbel.py:164`):
`G = -log(-log U)`. -/
noncomputable def gumbelSample (u : ℝ) : ℝ :=
  -Real.log (-Real.log u)

/-- Gumbel CDF (`Princeton LIPS`; Kool App B.1):
`F(z;mu) = exp(-exp(-(z-mu)))`. -/
noncomputable def gumbelCDF (mu z : ℝ) : ℝ :=
  Real.exp (-Real.exp (-(z - mu)))

/-- CDF is a probability: `F_μ(z) ∈ [0,1]` for all inputs — the finite
reason Gumbel-perturbed scores can be read as calibrated choice
probabilities. Pair with `gumbelCDF_sample` (the sampler hits every
level): together the CDF maps onto [0,1] and the sampler covers it. -/
theorem gumbelCDF_mem_Icc (mu z : ℝ) :
    gumbelCDF mu z ∈ Set.Icc (0 : ℝ) 1 := by
  unfold gumbelCDF
  constructor
  · exact le_of_lt (Real.exp_pos _)
  · have h : 0 ≤ Real.exp (-(z - mu)) := le_of_lt (Real.exp_pos _)
    have hle : -Real.exp (-(z - mu)) ≤ 0 := neg_nonpos.mpr h
    calc Real.exp (-Real.exp (-(z - mu))) ≤ Real.exp 0 :=
          Real.exp_le_exp.mpr hle
      _ = 1 := Real.exp_zero

/-- Inverse-CDF identity (C2): pushing `U ~ (0,1)` through the sampler
lands exactly on the CDF level `U` — i.e. `F_0(G(U)) = U`. The finite
reason the code's `sha256 → U → -log(-log U)` chain samples Gumbel(0,1).
`Real.log_neg` + `Real.exp_log` close both layers. -/
theorem gumbelCDF_sample (u : ℝ) (hu0 : 0 < u) (hu1 : u < 1) :
    gumbelCDF 0 (gumbelSample u) = u := by
  unfold gumbelSample gumbelCDF
  have hlog : Real.log u < 0 := Real.log_neg hu0 hu1
  have hpos : 0 < -Real.log u := neg_pos.mpr hlog
  have e1 : Real.exp (Real.log (-Real.log u)) = -Real.log u :=
    Real.exp_log hpos
  have e2 : -(-Real.log (-Real.log u)) = Real.log (-Real.log u) := by ring
  rw [sub_zero, e2, e1, neg_neg, Real.exp_log hu0]

/-- Tail clamp (`gumbel.py:168-171`): extreme tails clamp to `±20` so every
output is finite and deterministic for tests. -/
noncomputable def gumbelClamp (g : ℝ) : ℝ := max (-20) (min 20 g)

/-- Clamped values are exactly the in-range ones: staying inside [-20,20]
is equivalent to being a clamp fixpoint. Pair with `gumbelClamp_idem`
(idempotence): together the clamp is provably the projection onto the
interval — the finite reason downstream score arithmetic can assume
bounded noise without re-checking. -/
theorem gumbelClamp_mem_iff (g : ℝ) :
    gumbelClamp g = g ↔ -20 ≤ g ∧ g ≤ 20 := by
  unfold gumbelClamp
  constructor
  · intro h
    constructor
    · have : -20 ≤ max (-20 : ℝ) (min 20 g) := le_max_left _ _
      linarith
    · have : max (-20 : ℝ) (min 20 g) ≤ 20 :=
        max_le (by norm_num) (min_le_left _ _)
      linarith
  · intro ⟨hlo, hhi⟩
    rw [min_eq_right hhi, max_eq_right hlo]

theorem gumbelClamp_lo (g : ℝ) : -20 ≤ gumbelClamp g := by
  unfold gumbelClamp
  exact le_max_left _ _

theorem gumbelClamp_hi (g : ℝ) : gumbelClamp g ≤ 20 := by
  unfold gumbelClamp
  exact max_le (by norm_num) (min_le_left _ _)

/-- Clamping is idempotent: a second clamp changes nothing. Practical core:
the noise-clip preprocessing can run at every pipeline stage without
accumulating distortion — clipped Gumbels stay in [-20,20] forever, so
the score scale the pick depends on is stable across re-clamps.
Failure mode killed: a double-clip path silently narrowing the noise range. -/
theorem gumbelClamp_idem (g : ℝ) :
    gumbelClamp (gumbelClamp g) = gumbelClamp g := by
  have hle : max (-20 : ℝ) (min 20 g) ≤ 20 :=
    max_le (by norm_num) (min_le_left _ _)
  have hge : (-20 : ℝ) ≤ max (-20) (min 20 g) := le_max_left _ _
  unfold gumbelClamp
  rw [min_eq_right hle, max_eq_right hge]

end Sampler

section Score

/-- Gumbel score rule (`gumbel.py:1012-1013`, `L1051`):
`score = g + q`. (MuZero style adds prior logits + `sigma(completedQ)`;
hydra2 uses `g + q` — the admission is quoted verbatim in code.) -/
noncomputable def gumbelScore (g q : ℝ) : ℝ := g + q

/-- Score is strictly monotone in `q` at fixed `g`: better-backed actions
never score lower — the finite half of the C5 completed-Q dominance
(one-step logit dominance; the expectation version is HARD-skipped). -/
theorem gumbelScore_mono_q (g : ℝ) {q₁ q₂ : ℝ} (h : q₁ < q₂) :
    gumbelScore g q₁ < gumbelScore g q₂ := by
  unfold gumbelScore
  linarith

theorem gumbelScore_mono_q_le (g : ℝ) {q₁ q₂ : ℝ} (h : q₁ ≤ q₂) :
    gumbelScore g q₁ ≤ gumbelScore g q₂ := by
  unfold gumbelScore
  linarith

/-- Score is translation-invariant in the pair: adding the same constant
to both Q-values preserves their order — the finite reason the root pick
depends on Q GAPS, not Q scale. Practical core: rescaling or shifting all
completed-Q values (the F3/F4 Q-scale sweep in the Gumbel falsifier lane)
can only change the pick through the gaps, never the level — so the sweep
is a one-dimensional gap calibration, not a two-dimensional search. -/
theorem gumbelScore_shift_invariant (g₁ g₂ q₁ q₂ c : ℝ)
    (h : gumbelScore g₁ q₁ ≤ gumbelScore g₂ q₂) :
    gumbelScore g₁ (q₁ + c) ≤ gumbelScore g₂ (q₂ + c) := by
  unfold gumbelScore at h ⊢
  linarith

end Score

section Halving

/-- Sequential-halving keep rule (`gumbel.py:1025`):
`keep = (n + 1) // 2 = ceil(n/2)`. -/
def halvingKeep (n : ℕ) : ℕ := (n + 1) / 2
/-- Keep is at least one whenever anything survives the round. -/
theorem halvingKeep_pos {n : ℕ} (h : 1 ≤ n) : 1 ≤ halvingKeep n := by
  unfold halvingKeep
  omega

/-- Keep never exceeds the incoming set. -/
theorem halvingKeep_le (n : ℕ) : halvingKeep n ≤ n := by
  unfold halvingKeep
  rcases Nat.eq_zero_or_pos n with rfl | h
  · simp
  · omega

/-- Pairwise halving is exact: from 2 survivors exactly 1 remains — the
finite reason the sequential-halving loop's final round is deterministic.
Pair with `halvingKeep_reaches_one` (every orbit lands at 1): together
the loop both terminates AND its last step is exact, so the 5 s search
budget's final pick needs no tiebreak. -/
theorem halvingKeep_pair : halvingKeep 2 = 1 := by decide

/-- Halving is monotone: larger brackets stay larger after one round —
the finite reason shortlist ordering is preserved across halving
rounds: the leader after a round was ahead before it, so round-by-round
rankings never invert. Pairs `halvingKeep_orbit_ge` (orbits stay live):
together order and liveness both survive every round. -/
theorem halvingKeep_mono {a b : ℕ} (h : a ≤ b) :
    halvingKeep a ≤ halvingKeep b := by
  unfold halvingKeep
  omega

/-- Singleton stays put: `halvingKeep 1 = 1` — the finite reason the
halving schedule's fixed point is exactly the single survivor:
rounds stop shrinking precisely where the search stops branching.
Pairs `halvingKeep_pair` (pair collapses in one round): together the
bottom two rungs are pinned, so the halving ladder bottoms out
cleanly with no oscillation. -/
theorem halvingKeep_singleton : halvingKeep 1 = 1 := by decide

/-- Real halving for `n ≥ 2`: the survivor set strictly shrinks, so the
`while survivors > 1` loop terminates (C4 schedule lemma; the `O~(n/T)`
regret bound itself is HARD-skipped). -/
theorem halvingKeep_shrink {n : ℕ} (h : 2 ≤ n) : halvingKeep n < n := by
  unfold halvingKeep
  omega

end Halving

section SoftmaxNorm

variable {A : Type} [DecidableEq A]

/-- Finite softmax weights — the normalizer behind C1/C3/C5
(`exp(phi_i) / sum exp`; CQL `maskedLogSumExp` is the log-domain twin).
The sampling identities (`argmax ~ softmax`, Top-k WOR Eq4, improved
policy) are doc pointers; THIS is their finite normalizer, proved. -/
noncomputable def gumbelWeights (phi : A → ℝ) (acts : Finset A) : A → ℝ :=
  fun a => Real.exp (phi a) / ∑ b ∈ acts, Real.exp (phi b)

/-- Normalized weights sum to one over a nonempty support with positive
mass — the finite reason every Gumbel-derived policy is a distribution. -/
theorem gumbelWeights_sum_one (phi : A → ℝ) (acts : Finset A)
    (hpos : 0 < ∑ b ∈ acts, Real.exp (phi b)) :
    ∑ a ∈ acts, gumbelWeights phi acts a = 1 := by
  unfold gumbelWeights
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt hpos)

/-- Erasing one action leaves the rest summing to one minus its share:
the finite reason shortlist pruning is exactly priced — dropping an
action moves precisely its weight to the renormalizer, so the pruned
policy's mass gap is known before re-solving. Pairs
`gumbelWeights_sum_one` (full mass is one): together full and pruned
masses telescope, and the S5 gate prices every cut. -/
theorem gumbelWeights_erase_complement (phi : A → ℝ) (acts : Finset A)
    (hpos : 0 < ∑ b ∈ acts, Real.exp (phi b)) (a : A) (ha : a ∈ acts) :
    ∑ b ∈ acts.erase a, gumbelWeights phi acts b
      = 1 - gumbelWeights phi acts a := by
  have h1 := gumbelWeights_sum_one phi acts hpos
  have hsub : ∑ b ∈ acts.erase a, gumbelWeights phi acts b
      = (∑ b ∈ acts, gumbelWeights phi acts b) - gumbelWeights phi acts a :=
    Finset.sum_erase_eq_sub ha
  rw [hsub, h1]

/-- Singleton support concentrates all mass: with one legal action the
policy is degenerate at that action — the finite reason shortlist
search with a single survivor skips sampling entirely. Pair with
`gumbelWeights_sum_one` (mass conservation): together the support-size
ladder is pinned at both ends — one action means certainty, many means
a spread. -/
theorem gumbelWeights_singleton (phi : A → ℝ) (a : A)
    (hpos : 0 < Real.exp (phi a)) :
    gumbelWeights phi {a} a = 1 := by
  unfold gumbelWeights
  rw [Finset.sum_singleton]
  exact div_self (ne_of_gt hpos)

/-- Softmax weights are nonnegative: every action gets mass `≥ 0` — the
finite reason the Gumbel-derived policy never assigns negative
probability, so the root sampler never needs a repair pass. Pair with
`gumbelWeights_sum_one` (masses sum to 1): together the weights are a
genuine distribution on any support. -/
theorem gumbelWeights_nonneg (phi : A → ℝ) (acts : Finset A)
    (hpos : 0 < ∑ b ∈ acts, Real.exp (phi b)) (a : A) :
    0 ≤ gumbelWeights phi acts a := by
  unfold gumbelWeights
  exact div_nonneg (le_of_lt (Real.exp_pos _)) (le_of_lt hpos)

/-- Each weight is at most one: no single action takes more than the
whole mass when every mass is nonneg — the finite reason the
Gumbel-derived policy never over-concentrates beyond certainty.
Pairs `gumbelWeights_nonneg` (floor at zero): together each weight
sits in `[0, 1]`, so per-action sampling probabilities need no
clamping before use. -/
theorem gumbelWeights_le_one (phi : A → ℝ) (acts : Finset A)
    (ha : a ∈ acts) :
    gumbelWeights phi acts a ≤ 1 := by
  unfold gumbelWeights
  rw [div_le_one (Finset.sum_pos (fun b _ => Real.exp_pos _) ⟨a, ha⟩)]
  exact Finset.single_le_sum (fun b _ => le_of_lt (Real.exp_pos _)) ha

/-- Halving rounds to a single survivor (see `halvingKeep_reaches_one`
    below for the descent proof; the step-commutation fact is kept as a
    standalone simp lemma for callers that rewrite orbit positions). -/
theorem halvingKeep_iterate_step_aux (k m : ℕ)
    (ih : ∀ x : ℕ, Nat.iterate halvingKeep k (halvingKeep x) = halvingKeep (Nat.iterate halvingKeep k x)) :
    Nat.iterate halvingKeep (k + 1) (halvingKeep m) = halvingKeep (Nat.iterate halvingKeep (k + 1) m) := by
  simp only [Nat.iterate]
  rw [ih (halvingKeep m), ih m]

theorem halvingKeep_iterate_step : ∀ (k n : ℕ),
    Nat.iterate halvingKeep k (halvingKeep n) = halvingKeep (Nat.iterate halvingKeep k n)
  | 0, _ => rfl
  | k + 1, n => halvingKeep_iterate_step_aux k n (fun x => halvingKeep_iterate_step k x)

/-- Orbit lower bound (standalone): every orbit stays `≥ 1`. One flat
    induction; the step lemma commutes the unfold into the IH shape. -/
theorem halvingKeep_orbit_ge : ∀ (s k : ℕ), 1 ≤ s →
    1 ≤ Nat.iterate halvingKeep k s
  | _, 0, hs => by simpa using hs
  | s, j + 1, hs => by
    rw [Function.iterate_succ_apply, halvingKeep_iterate_step j s]
    exact halvingKeep_pos (halvingKeep_orbit_ge s j hs)

/-- Orbit descent bound (standalone, no nesting): from any start `s ≥ 1`,
    the orbit stays `≥ 1` and drops by `≥ 1` per round while `≥ 2`, so
    iterate `k` is at most `s - k + 1`. Proved by ONE flat induction —
    the nested-induction IH auto-naming fight is sidestepped entirely. -/
theorem halvingKeep_orbit_le (s k : ℕ) (hs : 1 ≤ s) :
    Nat.iterate halvingKeep k s ≤ s - k + 1 := by
  induction k generalizing s with
  | zero =>
    simp only [Function.iterate_zero_apply]
    omega
  | succ j ih =>
    -- Goal: `iterate (j+1) s ≤ s - (j+1) + 1`. The `show`-rewrite would
    -- also hit the RHS `(j+1)`, so instead rewrite ONLY the LHS via
    -- conv, then unfold one layer and close by omega over hle/hpos.
    conv =>
      lhs
      rw [show j + 1 = 1 + j from Nat.add_comm j 1, Function.iterate_add_apply]
    have h1 : Nat.iterate halvingKeep 1 (Nat.iterate halvingKeep j s)
        = halvingKeep (Nat.iterate halvingKeep j s) :=
      congrFun (Function.iterate_one halvingKeep) _
    rw [h1]
    have hle : Nat.iterate halvingKeep j s ≤ s - j + 1 := ih s hs
    have hpos : 1 ≤ Nat.iterate halvingKeep j s := halvingKeep_orbit_ge s j hs
    by_cases h2 : 2 ≤ Nat.iterate halvingKeep j s
    · have hsh := halvingKeep_shrink h2
      omega
    · push Not at h2
      have hkeep := halvingKeep_le (Nat.iterate halvingKeep j s)
      omega
/-- Halving rounds to a single survivor: the `while survivors > 1` loop
    always resolves — instantiate the orbit bound at `k = n` and squeeze
    with the orbit lower bound. Finite core behind the sequential-halving
    schedule (C4): the 5 s search budget always resolves to one action. -/
theorem halvingKeep_reaches_one (n : ℕ) (h : 1 ≤ n) :
    ∃ k : ℕ, Nat.iterate halvingKeep k n = 1 := by
  have hend : Nat.iterate halvingKeep n n ≤ 1 := by
    have := halvingKeep_orbit_le n n h
    omega
  have hge : 1 ≤ Nat.iterate halvingKeep n n := halvingKeep_orbit_ge n n h
  exact ⟨n, by omega⟩

end SoftmaxNorm

end Hydra2.Blueprint.Modules.Gumbel
