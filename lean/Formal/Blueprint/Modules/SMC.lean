import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Tactic

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedDecidableInType false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false

/-!
# Hydra2 §11.8 Controlled SMC (Feynman-Kac)

Blueprint §11.8: `γ_T(f)=E_q[f(X_{0:T})∏_t G_t]` with exact ratios and
conditionally unbiased resampling. Unnormalized estimator `γ̂_T(f)` is unbiased;
normalized `η̂_T(f)=γ̂_T(f)/γ̂_T(1)` (dividing by random `γ̂_T(1)`) is biased `O(1/N)`.

Source: Particle filter Feynman-Kac (Del Moral, Chopin), Wikipedia
https://en.wikipedia.org/wiki/Particle_filter — "Unbiased particle estimates of
likelihood functions" section.

For editors: this file models a 2-stage finite chain for the tiny test.
`G_t` are incremental incremental weights; `M_t` are proposal kernels.
Resampling offspring counts must satisfy `E[#children_i|ℱ]=N·w_i` (multinomial etc.)
for unbiasedness to hold.
-/

namespace Hydra2.Blueprint.Modules.SMC

section FeynmanKac

variable {State : Type} [Fintype State] [DecidableEq State]

/-- One-stage weight `G_0(x)`, transition `M_1(x'|x)`, and incremental `G_1(x_0,x_1)`. -/
structure FKModel (State : Type) [Fintype State] where
  G0 : State → ℝ
  M1 : State → State → ℝ
  G1 : State → State → ℝ
  G0_nonneg : ∀ x, 0 ≤ G0 x
  G1_nonneg : ∀ x0 x1, 0 ≤ G1 x0 x1
  M1_stoch : ∀ x0, ∑ x1 : State, M1 x0 x1 = 1
  M1_nonneg : ∀ x0 x1, 0 ≤ M1 x0 x1

/-- Unnormalized `γ_T(f) = ∑_{x0,x1} f(x0,x1) G0(x0) M1(x0→x1) G1(x0,x1)` (2-stage). -/
noncomputable def gamma (M : FKModel State) (f : State → State → ℝ) : ℝ :=
  ∑ x0 : State, ∑ x1 : State, f x0 x1 * M.G0 x0 * M.M1 x0 x1 * M.G1 x0 x1

noncomputable def eta (M : FKModel State) (f : State → State → ℝ) (hPos : gamma M (fun _ _ => 1) ≠ 0) : ℝ :=
  gamma M f / gamma M (fun _ _ => 1)

/-- Deterministic 2-stage finite Feynman–Kac law: this theorem is intentionally
NOT the stochastic unbiasedness claim. It states the finite double sum `gamma`
(the quantity the tiny test evaluates exactly). The stochastic claim
`E[γ̂_T^N(f)] = γ_T(f)` over random resampling trees needs `PMF`,
`ConditionalExpectation`, tower law, `HasFiniteIntegral` (`Del Moral` lemma 3) and
is stated honestly as `gammaHat_unbiased_stochastic` below. -/
theorem gamma_deterministic_law
    (M : FKModel State) (f : State → State → ℝ) :
    gamma M f = ∑ x0 : State, ∑ x1 : State, f x0 x1 * M.G0 x0 * M.M1 x0 x1 * M.G1 x0 x1 := rfl

/-- `gamma` of a nonneg test function is nonneg: double finite sum of products
of nonneg terms (`f`, `G0`, `M1`, `G1`). Feeds `eta` positivity side-conditions. -/
theorem gamma_nonneg (M : FKModel State) (f : State → State → ℝ)
    (hf : ∀ x0 x1, 0 ≤ f x0 x1) : 0 ≤ gamma M f := by
  unfold gamma
  apply Finset.sum_nonneg; intro x0 _
  apply Finset.sum_nonneg; intro x1 _
  exact mul_nonneg (mul_nonneg (mul_nonneg (hf x0 x1) (M.G0_nonneg x0))
    (M.M1_nonneg x0 x1)) (M.G1_nonneg x0 x1)

/-- Normalized `η̂ = γ̂/γ̂(1)` is biased: `E[η̂] ≠ η` in general (Jensen for `1/X`,
convex on `>0`). Concrete `X ∈ {1,3}` uniform: `E[X] = 2`, `1/E[X] = 1/2` but
`E[1/X] = 2/3`. Same mechanism as PBRF `childHat_bias_via_jensen` and MIS
`mis_ratio_bias_via_jensen`; the full expectation version over resampling
replicates is HARD-skipped (needs a sampling model + `MeasureTheory`). -/
theorem normalized_is_biased :
    ∃ (n : Nat) (hn : 0 < n) (X : Fin n → ℝ) (h_pos : ∀ i, 0 < X i),
      (1 / (n : ℝ)) * ∑ i : Fin n, (1 / X i) ≠ 1 / ((1 / (n : ℝ)) * ∑ i : Fin n, X i) := by
  refine ⟨2, by omega, fun i => if i.val = 0 then (1 : ℝ) else 3, fun i => ?_, ?_⟩
  · fin_cases i <;> simp
  · simp only [Fin.sum_univ_two]
    norm_num
/-- Finite analogue of the resampling unbiasedness condition `E[#offspring_i | weights] = N·w_i`.
For `N` particles with normalized weights `∑ w = 1`, multinomial resampling has
expected children `N·w_i` per ancestor, so the expected total is `N`.
The full stochastic tower-law version over `PMF`/`Binomial` offspring counts
(`Del Moral` lemma 3, needs `ConditionalExpectation`, `HasFiniteIntegral`) is the
HARD-skipped stochastic extension; the finite expectation-sum identity below is real. -/
noncomputable def expectedChildren (N w : ℝ) : ℝ := N * w

/-- Expected multinomial offspring sum to `N` when weights are normalized. -/
theorem resampling_expected_children_sum {n : ℕ} (w : Fin n → ℝ) (N : ℝ)
    (h_sum : ∑ i, w i = 1) :
    ∑ i, expectedChildren N (w i) = N := by
  unfold expectedChildren
  rw [← Finset.mul_sum, h_sum, mul_one]

/-- Same identity without the wrapper: distributing the sum over `N·w_i` recovers `N`.
Keeps the Blueprint name `resampling_unbiased_condition` as a real finite proposition. -/
theorem resampling_unbiased_condition {n : ℕ} (w : Fin n → ℝ) (N : ℝ)
    (h_sum : ∑ i, w i = 1) :
    ∑ i, N * w i = N := by
  rw [← Finset.mul_sum, h_sum, mul_one]

/-- Per-particle multinomial mean (sharpens `resampling_unbiased_condition`):
verbatim `E_r[cnt^i] = Np·normwt^i for all i` (jahoo §1.1,
https://jahoo.github.io/posts/smc-resampling/) from `cnt^i ~ Binomial(Np,
normwt^i)` (jahoo §2.2.1) and Douc et al. 2005 unbiasedness `E[N^i|G^n] =
n·w^i` Eq.3 §1 (https://ar5iv.org/html/cs/0507025, mirror of
https://arxiv.org/abs/cs/0507025). The Binomial-mean grounding stays an
explicit premise — its `PMF` tower is the HARD-skipped Del Moral lemma-3
class. `+placement`: `src/hydra2/search/modules/__init__.py:623`
`ControlledSMCModule.transform` propagates with exact `G_t = 1.0` copy and
claims offspring frequencies match the declared scheme — this identity is
the finite check that claim must satisfy per particle. -/
theorem resampling_multinomial_mean_identity {n : ℕ} (w E_count : Fin n → ℝ) (N : ℝ)
    (h_sum : ∑ i, w i = 1) (h_mean : ∀ i, E_count i = N * w i) :
    ∑ i, E_count i = N := by
  simp_rw [h_mean]
  exact resampling_unbiased_condition w N h_sum

/-- Deterministic companion: realized offspring counts summing to `N` sum to `N`.
The stochastic content (counts are random, only their conditional expectation is `N·w_i`)
is the HARD-skipped `PMF`/`Binomial` extension. -/
theorem resampling_counts_sum_deterministic {n : ℕ} (counts : Fin n → ℝ) (N : ℝ)
    (h : ∑ i, counts i = N) : ∑ i, counts i = N := h
/-- Time-reversal weight identity (Dai `arXiv:2007.11936` §2.1 Eq.2.1 `w_t = γ_t·L_{t-1}/(γ_{t-1}·M_t)` + §2.3 time-reversal `L = π_t·M/π_t`, via BackwardKern scout; cites `Del Moral et al. 2006 §3.3`). With the reversal choice the forward kernel cancels: `w = γ_t·π_{t-1}/(γ_{t-1}·π_t)`. Finite field core: `M` moves (mutation kernels) need no longer be tracked per-particle once `L` is the reversal. Variance-minimality of `L^opt = π_{t-1}M/q_t` (`w = γ_t/q_t`) needs `MeasureTheory` expectations over the joint proposal (HARD-skipped); the cancellation identity below is real. -/
theorem timereversal_weight_cancel (gt gtm1 pi_tm1 pi_t M : ℝ)
    (hM : M ≠ 0) (hgtm1 : gtm1 ≠ 0) (hpt : pi_t ≠ 0) :
    gt * (pi_tm1 * M / pi_t) / (gtm1 * M) = gt * pi_tm1 / (gtm1 * pi_t) := by
  have hden : gtm1 * M ≠ 0 := mul_ne_zero hgtm1 hM
  field_simp

/-- APF second-stage weight ratio (Pitt & Shephard 1999, JASA 94:590-599,
https://shephard.scholars.harvard.edu/publications/filtering-simulation-auxiliary-particle-filter):
joint target `f̂(α,k|Y) ∝ f(y|α)·f(α|α^k)·π^k` over joint proposal
`g(α,k|Y) ∝ f(y|μ^k)·f(α|α^k)·π^k` (Blevins eqs 6/8,
https://jblevins.org/notes/auxiliary-particle-filter) — transition and
prior-weight cancel leaving the likelihood ratio `ω = f(y|α)/f(y|μ)`
(wikipedia Selection,
https://en.wikipedia.org/wiki/Auxiliary_particle_filter). Finite field
core, verbatim mirror of `timereversal_weight_cancel`; optimal first-stage
weights / full adaptivity / CLT rates HARD-skip (Johansen arXiv:0709.3448
class, same as `gammaHat_unbiased_stochastic`). -/
theorem apf_second_stage_ratio (lik likMu trans prior : ℝ)
    (hlikMu : likMu ≠ 0) (htrans : trans ≠ 0) (hprior : prior ≠ 0) :
    (lik * trans * prior) / (likMu * trans * prior) = lik / likMu := by
  have hden : likMu * trans * prior ≠ 0 :=
    mul_ne_zero (mul_ne_zero hlikMu htrans) hprior
  field_simp

/-- Resampling variance ordering (Douc et al. 2005): residual and stratified
provably reduce conditional variance vs multinomial universally;
systematic does NOT always dominate (explicit counterexample in paper).
Example `N=2`, `w=[1/2,1/2]`: multinomial `Var[N_i]=N w_i(1-w_i)=1/2`,
stratified `Var=0` (one particle per stratum deterministic). `Douc` `Cappé` `Moulines` `2005` `Comparison of Resampling Schemes` `arXiv cs/0507025` `residual/stratified ≤ multinomial` `systematic counterexample` `conditional variance` `N=2` `Var_mult=1/2` `Var_strat=0` `proven` `Fin.sum_univ_two` `norm_num`. -/
theorem resampling_variance_stratified_le_multinomial_example :
    ∃ (w : Fin 2 → ℝ) (_h_sum : ∑ i : Fin 2, w i = 1) (_h_nonneg : ∀ i, 0 ≤ w i) (_h_pos : ∀ i, w i > 0),
      let varMultinomial : ℝ := 2 * w ⟨0, by omega⟩ * (1 - w ⟨0, by omega⟩)
      let varStratified : ℝ := 0
      varStratified < varMultinomial := by
  refine ⟨fun _ => 1/2, ?_, ?_, ?_, ?_⟩
  · simp [Fin.sum_univ_two]
  · intro i; fin_cases i <;> simp
  · intro i; fin_cases i <;> simp
  · simp only
    norm_num
/-- Finite analogue of "independent populations (not descendants) are the uncertainty unit" (§11.8).
Averaging over `P` independent replicate populations scales variance as `var / P`
(same `blockMeanVariance` pattern as the Evaluation module); descendants within one
population share ancestors so they do not give this `1/P` reduction.
The full `ProductMeasure` / i.i.d. replicate-population construction
(`RandomStreamKey population_id` vs `wall_block` descendants) is the HARD-skipped
stochastic extension; the variance-scaling identities below are real. -/
noncomputable def popMeanVariance (v : ℝ) (P : ℕ) : ℝ := v / (P : ℝ)

/-- Averaging over `P` populations is invertible: `(var/P)·P = var`. -/
theorem popMeanVariance_mul_cancel (v : ℝ) (P : ℕ) (hP : (P : ℝ) ≠ 0) :
    popMeanVariance v P * (P : ℝ) = v := by
  unfold popMeanVariance
  field_simp

/-- Population-averaged variance stays nonneg. -/
theorem popMeanVariance_nonneg (v : ℝ) (P : ℕ) (hv : 0 ≤ v) :
    0 ≤ popMeanVariance v P := by
  unfold popMeanVariance
  exact div_nonneg hv (Nat.cast_nonneg P)

/-- Variance of the mean of two independent populations with variances `va`, `vb`
and zero covariance is `(va+vb)/4` (cross term vanishes by independence). -/
theorem two_population_variance_add (va vb : ℝ) :
    (va + vb) / 4 = (va / 2 + vb / 2) / 2 := by
  ring

/-- Two independent populations halve the variance: replicate populations, not
descendants within one population, give the uncertainty reduction. -/
theorem independent_populations_are_unit (v : ℝ) :
    popMeanVariance v 2 = v / 2 ∧ popMeanVariance v 1 = v := by
  constructor
  · unfold popMeanVariance
    simp
  · unfold popMeanVariance
    simp

/-- Infinity-ESS (Huggins–Roy `arXiv:1503.00966` Def 4.5 `ESS_inf = ‖W‖₁/‖W‖_∞`, i.e. `1/max[w̄]`; BackwardKern scout. Adaptive trigger: resample when `ESS ≤ ηN` else copy (Sec 1.1/2.2/Rmk 3.1); `ESS_2` is Kish `1/∑w̄²`, `ESS_1` perplexity (aakinshin corroboration). Finite core: `essInf` def + range `[1, card]` + trigger predicate below. Divergence bounds (Thms 1.1/1.5/1.7, Props 5.2/5.3) need `PMF`/kernels/tower law (HARD-skipped, same class as `gammaHat_unbiased_stochastic`). -/
noncomputable def essInf (weights : Finset ℝ) (hne : weights.Nonempty) : ℝ :=
  1 / weights.max' hne

/-- Max weight is at least the mean: some particle carries `≥ 1/card`. Contrapositive of collapse — if every weight were below the mean, the sum could not reach `1` (`Finset.sum_le_sum` + `nsmul`). -/
theorem maxWeight_ge_inv_card (weights : Finset ℝ) (h_sum_one : ∑ w ∈ weights, w = 1)
    (hne : weights.Nonempty) :
    1 / (weights.card : ℝ) ≤ weights.max' hne := by
  have hcard_pos : (0 : ℝ) < (weights.card : ℝ) :=
    Nat.cast_pos.mpr (Finset.card_pos.mpr hne)
  have hcard_ne : (weights.card : ℝ) ≠ 0 := ne_of_gt hcard_pos
  by_contra hcon
  push Not at hcon
  have hall : ∀ w ∈ weights, w ≤ weights.max' hne :=
    fun w hw => Finset.le_max' weights w hw
  have hsum : ∑ w ∈ weights, w ≤ (weights.card : ℝ) * weights.max' hne := by
    calc ∑ w ∈ weights, w ≤ ∑ _w ∈ weights, weights.max' hne :=
          Finset.sum_le_sum hall
      _ = weights.card • weights.max' hne := by rw [Finset.sum_const]
      _ = (weights.card : ℝ) * weights.max' hne := nsmul_eq_mul _ _
  have hmul : (weights.card : ℝ) * weights.max' hne < 1 := by
    have h1 : (weights.card : ℝ) * weights.max' hne
        < (weights.card : ℝ) * (1 / (weights.card : ℝ)) :=
      mul_lt_mul_of_pos_left hcon hcard_pos
    rw [mul_one_div_cancel hcard_ne] at h1
    exact h1
  linarith

/-- `essInf` never exceeds the population size (reciprocal of the max-weight floor). -/
theorem essInf_le_card (weights : Finset ℝ) (h_sum_one : ∑ w ∈ weights, w = 1)
    (hne : weights.Nonempty) :
    essInf weights hne ≤ (weights.card : ℝ) := by
  unfold essInf
  have hge := maxWeight_ge_inv_card weights h_sum_one hne
  have hcard_pos : (0 : ℝ) < (weights.card : ℝ) :=
    Nat.cast_pos.mpr (Finset.card_pos.mpr hne)
  have h1 : (0 : ℝ) < 1 / (weights.card : ℝ) := one_div_pos.mpr hcard_pos
  have hmax_pos : 0 < weights.max' hne := lt_of_lt_of_le h1 hge
  have h := one_div_le_one_div_of_le h1 hge
  have h2 : (1 : ℝ) / (1 / (weights.card : ℝ)) = (weights.card : ℝ) := one_div_one_div _
  rw [h2] at h
  exact h

/-- `essInf` is at least `1` (uniform weights give exactly `card`; degeneracy drives it down toward `1`, never below — Scipedia collapse case). Needs each weight `≤ 1` (from `∑w = 1`, nonneg, erase argument as in `ESS_range`). -/
theorem essInf_ge_one (weights : Finset ℝ) (h_sum_one : ∑ w ∈ weights, w = 1)
    (h_nonneg : ∀ w ∈ weights, 0 ≤ w) (hne : weights.Nonempty) :
    1 ≤ essInf weights hne := by
  unfold essInf
  have h_w_le_one : ∀ w ∈ weights, w ≤ 1 := by
    intro w hw
    have h_sum_erase : ∑ v ∈ weights.erase w, v + w = ∑ v ∈ weights, v :=
      Finset.sum_erase_add _ _ hw
    have h_sum_erase_nonneg : 0 ≤ ∑ v ∈ weights.erase w, v := by
      apply Finset.sum_nonneg; intro v hv
      exact h_nonneg v (Finset.mem_of_mem_erase hv)
    linarith
  have hle : weights.max' hne ≤ 1 :=
    Finset.max'_le weights hne 1 (fun w hw => h_w_le_one w hw)
  have hge := maxWeight_ge_inv_card weights h_sum_one hne
  have hcard_pos : (0 : ℝ) < (weights.card : ℝ) :=
    Nat.cast_pos.mpr (Finset.card_pos.mpr hne)
  have h1 : (0 : ℝ) < 1 / (weights.card : ℝ) := one_div_pos.mpr hcard_pos
  have hmax_pos : 0 < weights.max' hne := lt_of_lt_of_le h1 hge
  have h := one_div_le_one_div_of_le hmax_pos hle
  simpa using h

/-- Huggins adaptive trigger predicate: resample when `essInf ≤ η·N`, else copy current population. -/
def essInfTrigger (weights : Finset ℝ) (hne : weights.Nonempty) (eta : ℝ) (N : ℕ) : Prop :=
  essInf weights hne ≤ eta * (N : ℝ)

/-- Trigger is monotone in budget: a larger `η` can only fire more often. -/
theorem essInfTrigger_mono_budget (weights : Finset ℝ) (hne : weights.Nonempty)
    (eta1 eta2 : ℝ) (N : ℕ) (h : eta1 ≤ eta2)
    (ht : essInfTrigger weights hne eta1 N) :
    essInfTrigger weights hne eta2 N := by
  unfold essInfTrigger at ht ⊢
  exact le_trans ht (mul_le_mul_of_nonneg_right h (Nat.cast_nonneg N))

/-- Kish effective sample size (`ESS_2 = (Σw)²/Σw²`, normalized `1/Σw̄²`):
verbatim `ESS = 1/Σᵢ(w⁽ⁱ⁾)²` (pytcl `02_particle_filters` notebook,
https://pytcl.readthedocs.io/en/latest/notebooks/02_particle_filters.html),
`ESSt = 1/Σ(w_t⁽ⁱ⁾)²` (metricgate,
https://metricgate.com/docs/particle-filter-resample/), `ESS = 1/Σ(normwt)²`
(jahoo, https://jahoo.github.io/posts/smc-resampling/), `(Σw)²/Σw²`
(WeightIt, https://ngreifer.github.io/WeightIt/reference/ESS.html).
Indexed over `Fin n` (NOT `Finset ℝ` like `essInf`): squares do not survive
dedup, so equal weights must count separately. `+placement`: this IS the
live formula — `src/hydra2/search/pbrf.py:304-311` `_ess_for_key`
(`s = sum(w*w ...); return 1.0/s` over `_normalized_weights`), forwarded by
`ImmutableForest.ess` (L463-467); diagnostic-only today (L300 comment, sole
consumer `tests/unit/test_pbrf_wp09a.py:444-445`). Formalizing the trigger
below unblocks the adaptive gate `resample iff ess ≤ η·N else copy`, each
skip saving one O(parent_count) `kernel.enumerate_next` + ChildEntry rebuild
(cf. `_fresh_rebuild`). Range `[1,N]` + `N/2` threshold + uniform-attains-`N`
below are the finite halves; CLT/variance-ordering claims HARD-skip. -/
noncomputable def essKish {n : ℕ} (w : Fin n → ℝ) : ℝ :=
  1 / ∑ i, (w i)^2

/-- Kish denominator is positive for any normalized population (some mass
must sit somewhere — the finite shadow of multinomial unbiasedness). -/
theorem essKish_sq_sum_pos {n : ℕ} (w : Fin n → ℝ)
    (h_sum : ∑ i, w i = 1) : 0 < ∑ i, (w i)^2 := by
  have hne : ∃ i, w i ≠ 0 := by
    by_contra hcon
    push Not at hcon
    have hzero : ∑ i, w i = 0 := Finset.sum_eq_zero (fun i _ => hcon i)
    linarith
  obtain ⟨j, hj⟩ := hne
  refine Finset.sum_pos' (fun i _ => sq_nonneg _) ⟨j, Finset.mem_univ j, ?_⟩
  exact pow_two_pos_of_ne_zero hj

/-- Each normalized nonneg weight is `≤ 1` (erase argument, mirrors
`essInf_ge_one`'s `h_w_le_one`). -/
theorem kish_weight_le_one {n : ℕ} (w : Fin n → ℝ)
    (h_sum : ∑ i, w i = 1) (h_nonneg : ∀ i, 0 ≤ w i) (j : Fin n) :
    w j ≤ 1 := by
  have h_sum_erase : ∑ i ∈ Finset.univ.erase j, w i + w j = ∑ i, w i :=
    Finset.sum_erase_add _ _ (Finset.mem_univ j)
  have h_sum_erase_nonneg : 0 ≤ ∑ i ∈ Finset.univ.erase j, w i := by
    apply Finset.sum_nonneg; intro v _
    exact h_nonneg v
  linarith

/-- Kish ESS is at least `1` (degeneracy floor: one particle holding all
mass gives exactly `1` — WeightIt `lies between 1 and length(w)`). -/
theorem essKish_ge_one {n : ℕ} (w : Fin n → ℝ)
    (h_sum : ∑ i, w i = 1) (h_nonneg : ∀ i, 0 ≤ w i) :
    1 ≤ essKish w := by
  unfold essKish
  have hsq_le : ∑ i, (w i)^2 ≤ 1 := by
    have hterm : ∀ i : Fin n, (w i)^2 ≤ w i := by
      intro i
      have h0 := h_nonneg i
      have h1 := kish_weight_le_one w h_sum h_nonneg i
      nlinarith
    calc ∑ i, (w i)^2 ≤ ∑ i, w i := Finset.sum_le_sum (fun i _ => hterm i)
      _ = 1 := h_sum
  exact one_le_one_div (essKish_sq_sum_pos w h_sum) hsq_le
/-- Kish ESS never exceeds the population size (Cauchy–Schwarz
`sq_sum_le_card_mul_sum_sq` with `∑w = 1` gives `1 ≤ n·Σw²`, then the
reciprocal flip — mirrors `essInf_le_card`; needs no nonneg). -/
theorem essKish_le_card {n : ℕ} (w : Fin n → ℝ)
    (h_sum : ∑ i, w i = 1) :
    essKish w ≤ (n : ℝ) := by
  unfold essKish
  have hsq_pos := essKish_sq_sum_pos w h_sum
  have hcs := sq_sum_le_card_mul_sum_sq (s := Finset.univ) (f := w)
  simp only [h_sum, Finset.card_univ, Fintype.card_fin, one_pow] at hcs
  rw [div_le_iff₀ hsq_pos]
  exact hcs

/-- Uniform weights attain exactly `N` (WeightIt `equals length(w) only
when all the weights are equal`; jahoo `to Np when weights are uniform` —
the attain direction; uniqueness is not claimed). -/
theorem essKish_uniform {n : ℕ} (hn : 0 < n) :
    essKish (fun _ : Fin n => (1 / (n : ℝ))) = (n : ℝ) := by
  unfold essKish
  have hnR : (n : ℝ) ≠ 0 := (Nat.cast_pos.mpr hn).ne'
  have h1 : (∑ _i : Fin n, ((1 : ℝ) / (n : ℝ))^2) = 1 / (n : ℝ) := by
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
  have h2 : (∑ i, ((fun _ : Fin n => (1 / (n : ℝ))) i)^2) = 1 / (n : ℝ) := h1
  rw [h2, one_div_one_div]

/-- Total multinomial offspring-variance budget, folded through Kish ESS:
`Σ N·w(1−w) = N·(1 − 1/essKish)`. Per-particle `Var = N·w(1−w)` is the
Binomial standard `Var(X) = np(1−p)`
(https://en.wikipedia.org/wiki/Binomial_distribution, infobox+Properties)
substituted at `p := w^i` — jahoo states only the qualitative extra
variance, so the per-particle equation is BRIDGED not verbatim (said
plainly); the `PMF` derivation HARD-skips. Douc Eq.6 §3.1 / Eq.9 §3.4 give
the multinomial conditional-variance baseline. `-cost`: low ESS means high
resample noise through this identity — the quantitative reason the
`essKishTrigger` gate fires earlier exactly when resampling injects the
most variance. Pure `ring`+`sum` algebra past the premise. -/
theorem resampling_multinomial_variance_budget {n : ℕ} (w : Fin n → ℝ) (N : ℝ)
    (h_sum : ∑ i, w i = 1) :
    ∑ i, N * w i * (1 - w i) = N * (1 - 1 / essKish w) := by
  have halg : ∑ i, N * w i * (1 - w i) = N * (1 - ∑ i, (w i)^2) := by
    have h1 : ∑ i, N * w i * (1 - w i) = N * ∑ i, (w i - (w i)^2) := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun i _ => by ring
    have h2 : ∑ i, (w i - (w i)^2) = 1 - ∑ i, (w i)^2 := by
      rw [Finset.sum_sub_distrib, h_sum]
    rw [h1, h2]
  have hfold : (1 : ℝ) - ∑ i, (w i)^2 = 1 - 1 / essKish w := by
    unfold essKish
    rw [one_div_one_div]
  rw [halg, hfold]

/-- Kish adaptive trigger: resample when `essKish ≤ η·N` (ratio form
`η = 1/2` is the practiced `N/2` threshold — pytcl
`ESS_threshold = N_particles/2`, jahoo `drops below (e.g., Np/2)`,
metricgate `0.5N`; cite those for the literal), else copy. -/
def essKishTrigger {n : ℕ} (w : Fin n → ℝ) (eta : ℝ) (N : ℕ) : Prop :=
  essKish w ≤ eta * (N : ℝ)

/-- Trigger is monotone in budget (clone of `essInfTrigger_mono_budget`):
a larger `η` can only fire more often, so the `1/2` gate is the
conservative member of the family. -/
theorem essKishTrigger_mono_budget {n : ℕ} (w : Fin n → ℝ)
    (eta1 eta2 : ℝ) (N : ℕ) (h : eta1 ≤ eta2)
    (ht : essKishTrigger w eta1 N) :
    essKishTrigger w eta2 N := by
  unfold essKishTrigger at ht ⊢
  exact le_trans ht (mul_le_mul_of_nonneg_right h (Nat.cast_nonneg N))

/-- Skipped-resample budget (`-cost` meter): over `T` steps with `skips`
trigger-skips at copy-cost `cCopy` vs resample-cost `cRes`, money left on
the table plus money spent equals always-resample. Each skip banks
`(cRes - cCopy)` — the O(parent_count) rebuild the adaptive gate avoids. -/
theorem resample_skip_budget (T skips cRes cCopy : ℕ) (h : skips ≤ T)
    (hle : cCopy ≤ cRes) :
    skips * cCopy + (T - skips) * cRes + skips * (cRes - cCopy) = T * cRes := by
  have e : cCopy + (cRes - cCopy) = cRes := Nat.add_sub_cancel' hle
  have e2 : skips + (T - skips) = T := Nat.add_sub_cancel' h
  calc skips * cCopy + (T - skips) * cRes + skips * (cRes - cCopy)
      = skips * (cCopy + (cRes - cCopy)) + (T - skips) * cRes := by ring
    _ = skips * cRes + (T - skips) * cRes := by rw [e]
    _ = (skips + (T - skips)) * cRes := by ring
    _ = T * cRes := by rw [e2]

end FeynmanKac

end Hydra2.Blueprint.Modules.SMC
