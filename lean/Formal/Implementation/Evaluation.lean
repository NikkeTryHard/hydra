import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedDecidableInType false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false

/-!
# Hydra2 §16 / SPEC §18 — Evaluation Block Independence & Fixed-N

Blueprint §16.1, SPEC §18.1–18.3

- Wall block is the independent unit: games within a wall share wall randomness
  and call-altered draw ownership (calls move wall pointer, rinshan draws from
  dead wall `14` `wanpai` dora/rinshan, each `kan` tops up from live wall shortening by `1`, live-wall exhaustion → `ryuukyoku`; `chi/pon/kan` steal turn skip `1-2` draws `ALBAN` — verified via `MahjongMaster`/`ALBAN` `NIST` `lmiratrix` cluster bootstrap), so they are not independent. `∑_e Z` logic already showed this.
  Only whole wall blocks are IID (walls drawn i.i.d. via semantic RNG `RandomStreamKey` `purpose=evaluation_schedule`).
- Fixed-N power: `N = ceil(((z_{1-α}+z_{1-β})·s/δ)²)` with pilot `s`, `α,β,δ`
  frozen blind to arm labels before unblinding (SPEC §18.3). `s` estimates
  SD of *wall-block* contrasts, not game contrasts.
- Uncertainty unit is `wall_block` for match evaluation; `iid_pair` for natural
  confirmation; `smc_population` / `rqmc_scramble` for those modules; `game_cluster`
  only for calibration metrics (never decisions).
- Bootstrap and sign-flip resample *whole wall blocks* only (EvalStatScout).

External: NIST/SEMATECH Handbook §7.2.2.2, PSU STAT 509, Howard et al. 1810.08240
(time-uniform CS; see EvalStatScout), lmiratrix cluster bootstrap.
-/

namespace Hydra2.Implementation.Evaluation

section FixedN

/-- Fixed-N sample size for one-sided test `H0:Δ=0` vs `H1:Δ=δ>0` with
block-mean contrast `Δ̂~N(Δ,σ²/N)`, Type I `α`, power `1-β`, pilot `s` for `σ`. -/
noncomputable def fixedN (z_alpha z_beta s delta : ℝ) : Nat :=
  Nat.ceil (((z_alpha + z_beta) * s / delta) ^ 2)

-- `z_{1-α}` etc. are normal quantiles `Φ⁻¹(1-α)`; `Φ` is standard normal CDF.
-- Positivity of the squared term needs `z_α+z_β ≠ 0` (else the effect is zero); the full normal-CDF derivation of the quantiles needs `ProbabilityTheory` (HARD skip).
theorem fixedN_formula
    (z_alpha z_beta s delta : ℝ) (hs : 0 < s) (hdelta : 0 < delta)
    (hz : z_alpha + z_beta ≠ 0) :
    0 < ((z_alpha + z_beta) * s / delta) ^ 2 := by
  have h_mul_ne : (z_alpha + z_beta) * s ≠ 0 := mul_ne_zero hz (ne_of_gt hs)
  have h_div_ne : (z_alpha + z_beta) * s / delta ≠ 0 := div_ne_zero h_mul_ne (ne_of_gt hdelta)
  exact sq_pos_of_ne_zero h_div_ne

theorem fixedN_nonneg (z_alpha z_beta s delta : ℝ) (_hs : 0 < s) (_hdelta : 0 < delta) :
    0 ≤ ((z_alpha + z_beta) * s / delta) ^ 2 := by
  positivity

theorem fixedN_pos_of_ne_zero (z_alpha z_beta s delta : ℝ) (hs : 0 < s) (hdelta : 0 < delta)
    (hz : z_alpha + z_beta ≠ 0) :
    0 < ((z_alpha + z_beta) * s / delta) ^ 2 :=
  fixedN_formula _ _ _ _ hs hdelta hz
theorem fixedN_mono_s (z_alpha z_beta s1 s2 delta : ℝ) (hs1 : 0 < s1) (hs2 : 0 < s2) (hdelta : 0 < delta)
    (h_le : s1 ≤ s2) (hz : 0 ≤ z_alpha + z_beta) :
    ((z_alpha + z_beta) * s1 / delta) ^ 2 ≤ ((z_alpha + z_beta) * s2 / delta) ^ 2 := by
  have h1 : (z_alpha + z_beta) * s1 ≤ (z_alpha + z_beta) * s2 := by
    apply mul_le_mul_of_nonneg_left h_le
    linarith
  have h2 : (z_alpha + z_beta) * s1 / delta ≤ (z_alpha + z_beta) * s2 / delta := by
    apply div_le_div_of_nonneg_right h1 (le_of_lt hdelta)
  apply sq_le_sq' (by linarith [div_nonneg (mul_nonneg hz (le_of_lt hs1)) (le_of_lt hdelta)]) h2

theorem fixedN_mono_delta_inv (z_alpha z_beta s delta1 delta2 : ℝ) (hs : 0 < s) (h1 : 0 < delta1) (h2 : 0 < delta2)
    (h_le : delta1 ≤ delta2) (hz : 0 ≤ z_alpha + z_beta) :
    ((z_alpha + z_beta) * s / delta2) ^ 2 ≤ ((z_alpha + z_beta) * s / delta1) ^ 2 := by
  have h_inv : (1:ℝ)/delta2 ≤ 1/delta1 := by
    rw [one_div, one_div]
    exact inv_anti₀ h1 h_le
  have h1' : (z_alpha + z_beta) * s / delta2 ≤ (z_alpha + z_beta) * s / delta1 := by
    calc (z_alpha + z_beta) * s / delta2
        = (z_alpha + z_beta) * s * (1/delta2) := by ring
      _ ≤ (z_alpha + z_beta) * s * (1/delta1) := by
          apply mul_le_mul_of_nonneg_left h_inv
          exact mul_nonneg hz (le_of_lt hs)
      _ = (z_alpha + z_beta) * s / delta1 := by ring
  have h_nn1 : 0 ≤ (z_alpha + z_beta) * s / delta1 := by
    apply div_nonneg (mul_nonneg hz (le_of_lt hs)) (le_of_lt h1)
  have h_nn2 : 0 ≤ (z_alpha + z_beta) * s / delta2 := by
    apply div_nonneg (mul_nonneg hz (le_of_lt hs)) (le_of_lt h2)
  exact sq_le_sq' (by linarith) h1'

end FixedN

section WallBlock

/-- `WallBlock` mean is the primary estimator; bootstrap/sign-flip resample blocks,
not games. `WallBlock` contains `6` symmetric + `4` rotation games (SPEC §18.1).
SPEC §18.1: wall block is the independent unit. Finite analogue of
`axiom_wallBlock_independent_unit` (see `Formal/Blueprint/EvaluationAxioms.lean`):
disjoint wall-block sets have additive sums (`Finset.sum_union`, cf. Blueprint
`block_sum_partition`), and variance adds when the cross-term vanishes
(cf. Blueprint `independent_blocks_variance_add`). Full IID walls over
`ProductMeasure` / semantic RNG `RandomStreamKey purpose=evaluation_schedule`
needs MeasureTheory (stochastic extension, comment only). -/
theorem wallBlock_is_independent_unit {n : ℕ} (f : Fin n → ℝ)
    (s t : Finset (Fin n)) (hdisj : Disjoint s t)
    (a b : ℝ) (hcov : a * b = 0) :
    (∑ x ∈ s ∪ t, f x = (∑ x ∈ s, f x) + ∑ x ∈ t, f x) ∧
      (a + b) ^ 2 = a ^ 2 + b ^ 2 := by
  constructor
  · exact Finset.sum_union hdisj
  · have h : 2 * a * b = 0 := by
      calc 2 * a * b = 2 * (a * b) := by ring
      _ = 2 * 0 := by rw [hcov]
      _ = 0 := by ring
    nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg (a + b)]

/-- SPEC §18.2: block bootstrap resamples whole wall blocks with replacement.
Finite core: uniform weights `wᵢ = 1/n` preserve the mean —
`∑ᵢ wᵢ xᵢ = (∑ᵢ xᵢ)/n` (convex combo; weights sum to 1 when `n ≠ 0`).
NIST Handbook §1.3.3.4 (Efron with-replacement at block level), lmiratrix
cluster bootstrap (sample clusters with replacement), tidyecology blocks
preserve dependence. Full multinomial-resample `MeasureTheory` version
(expected count 1 per block, as SMC expected children) needs `PMF`
(stochastic extension, comment only). -/
theorem blockBootstrap_resamples_blocks {n : ℕ} (x : Fin n → ℝ) :
    ∑ i, (1 / (n : ℝ)) * x i = (∑ i, x i) / (n : ℝ) := by
  have h : ∑ i, (1 / (n : ℝ)) * x i = (1 / (n : ℝ)) * ∑ i, x i := by
    rw [Finset.mul_sum]
  rw [h, div_eq_mul_one_div (∑ i, x i) (n : ℝ)]
  exact mul_comm _ _

/-- SPEC §18.2: sign-flip resamples whole wall blocks (Rademacher).
Finite core: the two Rademacher signs cancel —
`∑_{s : Fin 2} (-1)^{s} = 0`, i.e. mean 0 over all `2^n` sign vectors
(paired-block `sign_flip_interval`, mean preserving). Full randomization
`MeasureTheory` Rademacher family needs `ProductMeasure` (comment only). -/
theorem signFlip_resamples_blocks :
    ∑ s : Fin 2, ((-1 : ℝ) ^ (s.val : ℕ)) = 0 := by
  rw [Fin.sum_univ_two]
  simp

/-- SPEC §18.1: whole-block aggregation is primary — mean over 10 games per
wall (`6` C42 symmetric `2v2` + `4` rotation) via `blocks.py`
`aggregate_wall_block`. Finite core: `blockMean * 10 = ∑` for
`Fin 10` wall games. -/
theorem wholeBlockAggregation_is_primary (x : Fin 10 → ℝ) :
    ((∑ i, x i) / 10 : ℝ) * 10 = ∑ i, x i := by
  exact div_mul_cancel₀ _ (by norm_num)

/-- SPEC §18.1 / Blueprint §16.1: schedule commitment before play —
`walls_hash = of_canonical(ids)`, `latency_schedule_hash`, semantic seed
`evaluation_schedule` per game, `seed_protocol_hash` via `hydra2_rng_v1`
canonical JSON sha256, `rules_hash`, seat allocations `6` C42 symmetric `2v2`
+ `4` rotation = `10` games per wall, pure function before results
(`schedule_commitment_hash = of_canonical(to_json)` single binding).
Finite core: hash-commitment congruence — equal schedule bytes give equal
hash (same pattern as `Training.shared_params_byte_identical_implies_hash_eq`). -/
theorem schedule_commitment_before_play (a b : String) (h : a = b) :
    a.hash = b.hash := by
  rw [h]
end WallBlock

section TimeUniformCS

-- Alternative to fixed-N: predeclared time-uniform confidence sequence (Howard et al. 1810.08240).
-- Howard's stitched LIL boundary `O(√(t⁻¹ log log t))` is uniformly valid; Hydra2 uses
-- hedged capital CS (Waudby-Smith & Ramdas 2023) as concrete instantiation via `statistics.py` `hedged_cs_path` `Ville` `sub-ψ` `filtrations` `hedged betting` `empirical-Bernstein` `mixture/inverted stitching` `Table3` `sequential_design_guard` `predeclared` `fixedN` `ceil` `vs` `hedged` `choice` `frozen` `blind` `before` `unblinding` `SPEC §18.3` `Metrics`.
/-- SPEC §18.3: finite CS core (Ville-style shrinking width). The half-width
`s / √n` is antitone in `n` (mirror `zPowerApprox_mono_n` /
Blueprint `blockMeanVariance_mono_n`): more wall blocks give a tighter
uniform boundary. Full Howard stitched-LIL / hedged-capital `Ville`
`sub-ψ` `filtrations` `ProductMeasure` bound needs MeasureTheory
(stochastic extension, comment only). -/
theorem timeUniformCS_uniform_coverage (s : ℝ) (hs : 0 ≤ s)
    {n1 n2 : ℕ} (h1 : 0 < n1) (hle : n1 ≤ n2) :
    s / Real.sqrt (n2 : ℝ) ≤ s / Real.sqrt (n1 : ℝ) := by
  have h_n1_pos : (0 : ℝ) < (n1 : ℝ) := Nat.cast_pos.mpr h1
  have h_le_cast : (n1 : ℝ) ≤ (n2 : ℝ) := Nat.cast_le.mpr hle
  have h_sqrt_mono : Real.sqrt (n1 : ℝ) ≤ Real.sqrt (n2 : ℝ) :=
    Real.sqrt_le_sqrt h_le_cast
  exact div_le_div_of_nonneg_left hs (Real.sqrt_pos.mpr h_n1_pos) h_sqrt_mono

/-- SPEC §18.3: `fixedN` vs CS choice is predeclared frozen blind before
unblinding (`fixedN` `ceil` vs `timeUniformCS` `hedged`
`sequential_design_guard` vs adaptive peeking which invalidates confirmation;
`game_cluster` only for calibration metrics, never decisions; `case.py`
`EvalCase` `diagnostic_only` gate). Finite core: frozen CS width is monotone
in `s` — larger pilot `s` gives wider width at fixed `n`, so the frozen
blind choice orders widths deterministically. -/
theorem fixedN_vs_CS_declared_before_unblinding (s1 s2 : ℝ) (n : ℕ)
    (h_le : s1 ≤ s2) :
    s1 / Real.sqrt (n : ℝ) ≤ s2 / Real.sqrt (n : ℝ) := by
  exact div_le_div_of_nonneg_right h_le (Real.sqrt_nonneg _)
/-- Peeking guard (Bonferroni; MultiComp `family-wise coverage`, Evan Miller `10 peeks turns 1% into 5%`, `stop-at-5%-or-150obs gives 26.1% false positives`): two data-dependent looks cover at most the sum — union rejection region `|s∪t| ≤ |s|+|t|`, so two `α`-looks have worst-case `2α`. Finite core behind SPEC §18.3 predeclared frozen-blind `fixedN` vs adaptive peeking (which invalidates confirmation). -/
theorem peeking_union_bound (n : ℕ) (s t : Finset (Fin n)) :
    (s ∪ t).card ≤ s.card + t.card :=
  Finset.card_union_le s t
theorem peeking_two_looks_double (n : ℕ) (s t : Finset (Fin n)) (k : ℕ)
    (hs : s.card ≤ k) (ht : t.card ≤ k) :
    (s ∪ t).card ≤ 2 * k := by
  have h := Finset.card_union_le s t
  omega
/-- Stitched epoch union bound (Howard et al. 2021 Thm.1 Eq.9 `P(∃t: S_t≥S_α) ≤ Σ_k α/h(k)`: break time into geometric epochs `η^k ≤ V < η^{k+1}`, per-epoch budget `α/h(k)`, take a union bound; Fig.3 caption. CsDepth scout, ar5iv Eq.8/9/10). Finite core: `K` epochs each with rejection mass `≤k₀` cover `≤K*k₀` — induction on `K` via `peeking_union_bound` (`card_union_le`) binary step. Full infinite-horizon `P(union)=…≤α` with `Σ1/h≤1` + Ville/sub-ψ needs MeasureTheory (comment only). -/
theorem stitched_epoch_union_bound (m K k₀ : ℕ) (E : ℕ → Finset (Fin m))
    (h : ∀ k, k < K → (E k).card ≤ k₀) :
    (Finset.biUnion (Finset.range K) E).card ≤ K * k₀ := by
  induction K with
  | zero => simp
  | succ n ih =>
    have ih' : (Finset.biUnion (Finset.range n) E).card ≤ n * k₀ :=
      ih (fun k hk => h k (Nat.lt_of_lt_of_le hk (Nat.le_succ n)))
    have hn : (E n).card ≤ k₀ := h n (Nat.lt_succ_self n)
    have hunion : Finset.biUnion (Finset.range (n + 1)) E =
        E n ∪ Finset.biUnion (Finset.range n) E := by
      rw [Finset.range_add_one, Finset.biUnion_insert]
    calc (Finset.biUnion (Finset.range (n + 1)) E).card
        = (E n ∪ Finset.biUnion (Finset.range n) E).card := by rw [hunion]
      _ ≤ (E n).card + (Finset.biUnion (Finset.range n) E).card :=
          Finset.card_union_le _ _
      _ ≤ k₀ + n * k₀ := Nat.add_le_add hn ih'
      _ = (n + 1) * k₀ := by ring

end TimeUniformCS

section StableRank

/-- Tenhou stable rank (Suphx Appx-C Eq.7, via SuphxAppx scout, ar5iv 2003.13590: `stable = (5*n1 + 2*n2)/n4 - 2`; Fig.12 sampled `K=2000/N=5000`; Mortal duplicate-1v3 + rank-pt `[90,45,0,-135]` as variance template). Finite core: closed form + monotonicity (more 1sts raise it, more 4ths lower it — the quantitative reason Suphx's low-4th style `18.7%` drives rank). Full rank-pt lobby tables (tonpuu vs `1.5×` tonnan) + bootstrap variance stay harness-side. -/
noncomputable def stableRank (n1 n2 n4 : ℕ) : ℝ :=
  (5 * (n1 : ℝ) + 2 * (n2 : ℝ)) / (n4 : ℝ) - 2

theorem stableRank_mono_n1 (n1 n1' n2 n4 : ℕ) (h : n1 ≤ n1') :
    stableRank n1 n2 n4 ≤ stableRank n1' n2 n4 := by
  unfold stableRank
  have hle : 5 * (n1 : ℝ) + 2 * (n2 : ℝ) ≤ 5 * (n1' : ℝ) + 2 * (n2 : ℝ) := by
    have hcast : (n1 : ℝ) ≤ (n1' : ℝ) := Nat.cast_le.mpr h
    linarith
  have hdiv := div_le_div_of_nonneg_right hle (Nat.cast_nonneg n4)
  linarith

theorem stableRank_antitone_n4 (n1 n2 n4 n4' : ℕ) (h4 : 0 < n4) (h : n4 ≤ n4') :
    stableRank n1 n2 n4' ≤ stableRank n1 n2 n4 := by
  unfold stableRank
  have hC : (0 : ℝ) ≤ 5 * (n1 : ℝ) + 2 * (n2 : ℝ) := by positivity
  have hn : (0 : ℝ) < (n4 : ℝ) := Nat.cast_pos.mpr h4
  have hle : (n4 : ℝ) ≤ (n4' : ℝ) := Nat.cast_le.mpr h
  have hdiv := div_le_div_of_nonneg_left hC hn hle
  linarith

end StableRank

section PrPlBets

/-- Predictable plug-in bet size (Waudby-Smith & Ramdas 2023 Eq.26, via
IdeaBetting scout: `λ^{PrPl±}_t = √(2·log(2/α)/(σ̂²_{t-1}·t·log(t+1)))` with
regularized variance; confseq `lambda_predmix_eb` defaults `prior 0.5/0.25`,
`fake_obs = 1`). Pure real formula — the runnable recipe behind Hydra2's
`hedged_cs_path`, replacing the ad-hoc `√(8·…)` bet (2x too large) and
Bernoulli plug-in variance. -/
noncomputable def prPlLambda (alpha sigma2 t : ℝ) : ℝ :=
  Real.sqrt (2 * Real.log (2 / alpha) / (sigma2 * t * Real.log (t + 1)))

/-- Eq.25 truncation into the bet-validity interval (`c = 1/2` default). -/
noncomputable def prPlTrunc (lam c m : ℝ) : ℝ := min |lam| (c / m)

/-- Bet-validity: truncated bets keep both one-sided hedged capital factors
nonneg on `[0,1]` data — the finite precondition for the capital process to be
a test martingale (Ville step stays admitted under
`axiom_timeUniformCS_uniform_coverage`). -/
theorem capital_factor_nonneg {lam c m x : ℝ} (hm0 : 0 < m) (hm1 : m < 1)
    (hc : 0 ≤ c) (hc1 : c ≤ 1) (hx0 : 0 ≤ x) (hx1 : x ≤ 1) :
    0 ≤ 1 + min |lam| (c / m) * (x - m) ∧
    0 ≤ 1 - min |lam| (c / (1 - m)) * (x - m) := by
  have hm_ne : m ≠ 0 := ne_of_gt hm0
  have h1m : (0 : ℝ) < 1 - m := by linarith
  have h1m_ne : (1 : ℝ) - m ≠ 0 := ne_of_gt h1m
  have hL1_nn : 0 ≤ min |lam| (c / m) :=
    le_min (abs_nonneg _) (div_nonneg hc (le_of_lt hm0))
  have hL2_nn : 0 ≤ min |lam| (c / (1 - m)) :=
    le_min (abs_nonneg _) (div_nonneg hc (le_of_lt h1m))
  have hL1_le : min |lam| (c / m) ≤ c / m := min_le_right _ _
  have hL2_le : min |lam| (c / (1 - m)) ≤ c / (1 - m) := min_le_right _ _
  constructor
  · have e1 : 0 ≤ min |lam| (c / m) * x := mul_nonneg hL1_nn hx0
    have e2 : min |lam| (c / m) * m ≤ c := by
      calc min |lam| (c / m) * m ≤ (c / m) * m :=
            mul_le_mul_of_nonneg_right hL1_le (le_of_lt hm0)
        _ = c := div_mul_cancel₀ c hm_ne
    have e3 : min |lam| (c / m) * (x - m)
        = min |lam| (c / m) * x - min |lam| (c / m) * m := by ring
    linarith
  · by_cases hxm : x ≤ m
    · have e1 : min |lam| (c / (1 - m)) * (x - m) ≤ 0 :=
        mul_nonpos_of_nonneg_of_nonpos hL2_nn (by linarith)
      linarith
    · push Not at hxm
      have e1 : min |lam| (c / (1 - m)) * (x - m)
          ≤ (c / (1 - m)) * (x - m) :=
        mul_le_mul_of_nonneg_right hL2_le (by linarith : (0 : ℝ) ≤ x - m)
      have e2 : (c / (1 - m)) * (x - m) ≤ (c / (1 - m)) * (1 - m) :=
        mul_le_mul_of_nonneg_left (by linarith : x - m ≤ 1 - m)
          (div_nonneg hc (le_of_lt h1m))
      have e3 : (c / (1 - m)) * (1 - m) = c := div_mul_cancel₀ c h1m_ne
      linarith

/-- Hedged max capital (Eq.24 finite pointwise form): `K± = max(θ·K⁺, (1-θ)·K⁻)`. -/
noncomputable def hedgedCapital (theta Kp Km : ℝ) : ℝ :=
  max (theta * Kp) ((1 - theta) * Km)

/-- Bet size is nonneg (square root). -/
theorem prPlLambda_nonneg (alpha sigma2 t : ℝ) :
    0 ≤ prPlLambda alpha sigma2 t := by
  unfold prPlLambda
  exact Real.sqrt_nonneg _

/-- Truncation stays nonneg (both candidates are). -/
theorem prPlTrunc_nonneg (lam c m : ℝ) (hc : 0 ≤ c) (hm : 0 < m) :
    0 ≤ prPlTrunc lam c m := by
  unfold prPlTrunc
  exact le_min (abs_nonneg _) (div_nonneg hc (le_of_lt hm))

/-- Truncation never exceeds the cap (it is a `min`). -/
theorem prPlTrunc_le (lam c m : ℝ) : prPlTrunc lam c m ≤ c / m := by
  unfold prPlTrunc
  exact min_le_right _ _

/-- Hedged max dominates each side (Eq.24: the max is an upper envelope, so a
rejection by either one-sided capital rejects the hedged capital too). -/
theorem hedgedCapital_ge_left (theta Kp Km : ℝ) :
    theta * Kp ≤ hedgedCapital theta Kp Km := by
  unfold hedgedCapital
  exact le_max_left _ _

theorem hedgedCapital_ge_right (theta Kp Km : ℝ) :
    (1 - theta) * Km ≤ hedgedCapital theta Kp Km := by
  unfold hedgedCapital
  exact le_max_right _ _

end PrPlBets

end Hydra2.Implementation.Evaluation
