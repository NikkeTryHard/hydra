import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Tactic

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedDecidableInType false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false

/-! # Hydra2 Gumbel search identities (Gumbel-Top-k + sequential halving)

Mirrors `src/hydra2/search/gumbel.py` (1819-line live planner, zero prior
Lean, zero `ideas/gumbel*`): deterministic inverse-CDF sampler
(`deterministic_gumbel` L133-172), root gumbels L175-190, score rule
`score = g + q` L1012-1013/L1051, halving keep `ceil(n/2)` L1024-1029,
final argmax L1044-1065.

Paper chain (GumbelMiner, Wave 1):
- C1 Gumbel-Max: `argmax_i{phi_i + G_i} ~ Categorical(softmax phi)`
  (Kool et al. ICML19 §2.3; derivation Princeton LIPS 2013).
  URLs: https://ar5iv.labs.arxiv.org/html/1903.06059
        https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
- C2 CDF `F(z;mu) = exp(-exp(-(z-mu)))` + sampler `G = phi - log(-log U)`
  (Kool §2.2, App B.1 Eq20-21).
- C3 Gumbel-Top-k Thm 1 §2.4: ordered top-k = ordered WOR sample
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

theorem gumbelClamp_lo (g : ℝ) : -20 ≤ gumbelClamp g := by
  unfold gumbelClamp
  exact le_max_left _ _

theorem gumbelClamp_hi (g : ℝ) : gumbelClamp g ≤ 20 := by
  unfold gumbelClamp
  exact max_le (by norm_num) (min_le_left _ _)

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

end SoftmaxNorm

end Hydra2.Blueprint.Modules.Gumbel
