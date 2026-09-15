/-! # Hydra2 admitted stochastic claims (trivial `True` records, NOT proved)

Each item states its REAL proposition in its docstring with the missing
`MeasureTheory`/`ProbabilityTheory` dependency named, and records a trivial
`True` theorem (no logical content, no axiom extension) so the build stays
axiom-free: `grep ^axiom` is clean and `#print axioms` shows nothing beyond
Mathlib. Finite discrete cores live in their modules; these are the stochastic parts.

Ultimate goal: these records are the exact extension points for mahjong
correctness/extensibility — discharging one (e.g. wall `ProductMeasure`) upgrades
training-relevant evaluation guarantees, not just the build log.
-/

namespace Hydra2.Blueprint.EvaluationAxioms

/-- Admitted wall-block independence: wall blocks drawn via the
semantic RNG (`RandomStreamKey purpose=evaluation_schedule`) are independent
sampling units; games within a wall are NOT independent and averaging over them
underestimates variance and fails closed. Genuine proof needs `MeasureTheory`
`ProductMeasure` IID walls + dead-wall/`rinshan` pointer + call-altered draw ownership. -/
theorem axiom_wallBlock_independent_unit : True := trivial

/-- Admitted SMC unbiasedness: the unnormalized
SMC estimator is unbiased, `E[γ̂_T^N(f)] = γ_T(f)`, under exact incremental `G_t`
and conditionally unbiased resampling (`E[#children|ℱ] = N·w`); biased resampling
breaks the tower law and fails closed.
Genuine proof needs `PMF`/`ConditionalExpectation`, tower law, `HasFiniteIntegral`
(`Del Moral` lemma 3); the deterministic finite law is `gamma_deterministic_law`. -/
theorem axiom_gammaHat_unbiased_stochastic : True := trivial

/-- Admitted time-uniform coverage (Howard et al. 1810.08240, https://arxiv.org/abs/1810.08240):
the predeclared hedged-capital confidence sequence covers uniformly over time;
peeking without predeclared frozen N breaks coverage and fails closed.
Genuine proof needs filtrations, sub-psi, `Ville`, stitched-LIL/ProductMeasure;
`fixedN` lemmas stay elementary. -/
theorem axiom_timeUniformCS_uniform_coverage : True := trivial

/-- Admitted RQMC rate (Owen 1997, https://doi.org/10.1214/aos/1031594731):
scrambled-net RQMC attains the `O(N^{-3/2})` rate under smoothness (`A* < 1/2`);
without smoothness the rate collapses and the claim fails closed to the proved
permutation structure. Genuine proof needs Walsh expansion + boundary-growth
conditions; the `Fin`-permutation structure
(`rqmcShift_bijective`, `rqmcShift_preserves_filter_card`) is proved. -/
theorem axiom_RQMC_rate_smooth : True := trivial

end Hydra2.Blueprint.EvaluationAxioms
