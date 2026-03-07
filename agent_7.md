## 1

**name**
Shift-aware support-constrained search distillation

**outside ingredients and sources**
SPIBB’s core idea is to improve safely by bootstrapping back to the baseline policy where support is weak; Soft-SPIBB relaxes that into local uncertainty-aware constraints; selective calibration/abstention adds the extra rule that it is often better to reject uncertain targets than to train on all of them. ([Proceedings of Machine Learning Research][1])

**why it transfers to Hydra specifically**
Hydra already wants ExIt, search-as-feature, CT-SMC belief, AFBS, and pondering; the reconciliation memo says the real bottleneck is that these loops are only partially closed, not that Hydra lacks advanced module names. The live encoder already has fixed search/belief/Hand-EV planes in a 192×34 superset, so Hydra can amortize trusted search outputs without an architecture reset. ([GitHub][2])

Hydra’s code is unusually ready for this: `MjaiSample` already carries `safety_residual`, `belief_fields`, and `mixture_weights`; `mjai_loader.rs` already builds and packs those targets; `HydraTargets.policy_target` is already dense; `soft_target_from_exit` already exists; `delta_q` and `safety_residual` heads already exist; and the loss already computes `l_delta_q`. But `sample.rs` still leaves `delta_q_target: None`, and the advanced loss weights default to `0.0`. That is almost the textbook case for “safe distillation of existing search,” not “do more search everywhere.” ([GitHub][3])

**exact Hydra surfaces it would touch**
`hydra-core/src/afbs.rs` for root support/trust stats, `hydra-core/src/bridge.rs` for optional trust/debug feature export, `hydra-train/src/data/sample.rs` for filling search-derived targets, `hydra-train/src/training/losses.rs` for turning on and weighting the existing losses, and `hydra-train/src/model.rs` only for consuming the heads it already has. [blocked]: whichever self-play/sample-writer path persists AFBS root stats is not in the supplied slice, so that one shim may still need to be found. ([GitHub][4])

**implementation sketch or pseudocode**
Use the existing hard-state gate from the prior Hydra answer as the first filter, then apply Soft-SPIBB-like support mixing only where search is actually backed by visits.

```text
hard_state =
    (top2_gap < 0.10) OR
    (max_risk > 0.15) OR
    (ess_ratio < 0.45) OR
    (wall <= 12)

if hard_state and root_visits >= 64:
    for legal action a:
        support_a = clip(visits[a] / 4, 0, 1)
        trust_a = support_a * calib(root_visits, ess_ratio, search_entropy, belief_entropy)
        delta_q_target[a] = trust_a * clip(Q[a] - V_root, -dq_max, dq_max)

    mix = max_a trust_a
    policy_target = soft_target_from_exit(model_probs, exit_policy, legal_mask, mix)
else:
    no search targets
```

Start with a fixed calibrator, not a learned one. The existing Hydra recommendation already gives usable thresholds (`top2_gap < 0.10`, `max risk > 0.15`, `ESS/P < 0.45`, `wall <= 12`, `root visits >= 64`, `supported-action visits >= 4`). ([GitHub][5])

**cheapest prototype path**
Do not invent a new head. Reuse the dense `policy_target`, populate `delta_q_target` in `sample.rs`, turn on `w_delta_q`, and only emit search targets on the hard-state slice above. Benchmark against an 8×-budget AFBS teacher before any full retrain. I also pressure-tested this locally with a small standalone harness against the real downloaded files: the hook check passed, and the gate logic is trivial to wire.

**what success would look like**
On held-out hard states, the accepted subset should show materially better action-match / KL agreement to an 8×-budget AFBS teacher than naïve “use every searched state,” at still-useful coverage. Then, with the same runtime budget, self-play should improve in Elo / average placement without needing deeper online search. ([GitHub][6])

**what would kill the idea quickly**
If production-budget AFBS is not consistently better than the base policy even on the accepted subset, or if accepted coverage is too small to matter, stop. The deeper hidden kill is systematic teacher bias: because Hand-EV is still not the more realistic oracle Hydra wants and endgame is still weighted PIMC rather than true exactification, this method could amplify the wrong search bias faster than it adds signal. ([GitHub][6])

## 2

**name**
Paired-scenario action racing for endgame exactification

**outside ingredients and sources**
Common random numbers reduce comparison variance when alternatives are evaluated on the same random scenarios; ranking-and-selection gives a sequential way to allocate simulations among contenders; empirical Bernstein stopping uses variance to stop earlier than fixed-budget racing. ([arXiv][7])

**why it transfers to Hydra specifically**
Hydra’s architecture explicitly wants selective endgame exactification and stronger Hand-EV, but the reconciliation memo says Hand-EV is still heuristic and endgame is still weighted PIMC rather than the stronger exactification path. In code, `bridge.rs` already has `compute_ct_smc_hand_ev`, and `endgame.rs` already has `pimc_endgame_q_topk` over top-mass particles. So the cheapest asymmetric win is not “more endgame search,” but “evaluate actions on the same hidden scenarios, then stop when the leader is statistically separated.” ([GitHub][2])

**exact Hydra surfaces it would touch**
`hydra-core/src/endgame.rs` first, `hydra-core/src/ct_smc.rs` only if you want a reusable joint-scenario sampler, and later `hydra-core/src/hand_ev.rs` [blocked] once Hydra has a slightly richer micro-rollout evaluator than the current heuristic offensive estimate. ([GitHub][4])

**implementation sketch or pseudocode**
Replace “estimate each action independently over top-mass particles” with “race the top candidates on the same scenario ids.”

```text
scenarios = draw_joint_scenarios_from_top_mass_particles(state_hash, particles, K)

for a in top_k_legal_actions:
    for s in scenarios[:m0]:
        y[a, s] = eval_suffix_under_same_hidden_world(a, s)

leader = argmax_a mean(y[a, :])
runner = second_best()

while budget_left:
    update paired diff d_s = y[leader, s] - y[runner, s]
    if EB_lower_bound(mean(d), var(d), n) > 0:
        break
    allocate next batch on NEW shared scenarios to leader and runner
    maybe replace runner if another action catches up
```

This is a compute-allocation change, not an architecture change. In a small toy simulation I ran here, shared-scenario pairing cut the standard deviation of the mean action-difference estimate by about **1.8×** and improved correct selection rate under the same sample count.

**cheapest prototype path**
Only patch `pimc_endgame_q_topk` for `wall <= 10`: same selected particle list for all actions, deterministic scenario seed from state hash, top-2 or top-3 discard candidates only, and an empirical-Bernstein stop on the leader vs runner-up. Do not touch AFBS yet.

**what success would look like**
At the same wall-clock budget, action choice should become more stable across reruns, agreement with a much higher-budget endgame evaluator should improve, and self-play uplift should concentrate in the `wall <= 10` slice. ([GitHub][2])

**what would kill the idea quickly**
If action-conditioned trajectories decorrelate so fast that pairing does not actually induce useful positive correlation, or if the current suffix evaluator is too crude for lower-variance comparisons to matter, stop. This dies fast if the variance reduction is real but the bias is still dominant.

## 3

**name**
Value-of-computation ponder control

**outside ingredients and sources**
Russell–Wefald metareasoning treats computation as an action whose value is the expected improvement in decision quality; later MCTS metareasoning work argues that computation control is better viewed as a ranking-and-selection problem than a plain visit-allocation heuristic. ([IJCAI][8])

**why it transfers to Hydra specifically**
Hydra Final already treats opponent-turn idle time and predictive pondering as a first-class edge. In code, `afbs.rs` already has `PonderResult`, `PonderCache`, predicted-child caching, and a current priority rule that is just `(0.1 - top2_gap)+risk+(1-ESS)`. That is exactly the kind of heuristic that VOC control can upgrade without increasing total search. ([GitHub][2])

**exact Hydra surfaces it would touch**
Almost entirely `hydra-core/src/afbs.rs`; optionally a tiny telemetry hook in `bridge.rs` if you want to log cache reuse and action-flip events for fitting the scheduler. [blocked]: if current runtime does not already emit AFBS trace data, you need one light logging shim before fitting the scheduler. ([GitHub][9])

**implementation sketch or pseudocode**
Replace static priority with estimated value-per-millisecond.

```text
for queued root r:
    evoc(r) =
        P(action_flip after next chunk | gap, visits, ess, risk, depth)
        * abs(Q1 - Q2)
        * P(predicted_child_cache_reused before expiry)

    priority(r) = evoc(r) / expected_ms(next_chunk)

    stop pondering r when:
        CI(best - runner_up) is separated
        OR evoc(r) < epsilon
```

First pass: fit `P(action_flip)` with a tiny logistic model from AFBS trace data; no neural change, no extra head, no broader search.

**cheapest prototype path**
Keep the existing queue and cache; only replace `compute_ponder_priority` with an offline-calibrated formula using current features (`top2_gap`, `risk_score`, `particle_ess`, `visit_count`, maybe cache freshness). Then compare the same total ponder budget against the current heuristic. ([GitHub][9])

**what success would look like**
More cache hits that actually change downstream root decisions, less wasted pondering on already-settled roots, and better self-play at equal total ponder compute. That is a pure compute-reuse win, not a bigger-search win. ([GitHub][2])

**what would kill the idea quickly**
If the game rarely reuses predicted-child states, or extra pondering almost never flips the chosen action, VOC control has nothing to optimize and the current cheap heuristic is already enough.

**the single best candidate to try first**
Shift-aware support-constrained search distillation.

**the single best cheap benchmark to run first**
Take a held-out hard-state set using the existing Hydra triggers (`top2_gap < 0.10` or `risk > 0.15` or `ESS/P < 0.45` or `wall <= 12`), run production-budget AFBS and 8×-budget AFBS, and compare naïve-all-state distillation vs trust-gated distillation on accepted-state action match / KL to the 8× teacher plus accepted coverage. If the gated subset is not clearly better at useful coverage, do not train.

**the single biggest hidden risk in your recommendation**
You may distill current AFBS bias faster than you distill useful search signal, because Hydra’s present search teacher still sits on a not-yet-realistic Hand-EV and a still-simplified endgame evaluator.

[1]: https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf "https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf"
[2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md"
[3]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/sample.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/sample.rs"
[4]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs"
[5]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md"
[6]: https://raw.githubusercontent.com/NikkeTryHard/hydra/reconcile-hydra-plan/research/design/HYDRA_RECONCILIATION.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/reconcile-hydra-plan/research/design/HYDRA_RECONCILIATION.md"
[7]: https://arxiv.org/pdf/1410.6782 "https://arxiv.org/pdf/1410.6782"
[8]: https://www.ijcai.org/Proceedings/89-1/Papers/053.pdf "https://www.ijcai.org/Proceedings/89-1/Papers/053.pdf"
[9]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/afbs.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/afbs.rs"
