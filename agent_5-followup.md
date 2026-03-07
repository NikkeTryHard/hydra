1. what Hydra already has that matters for this question

Hydra is already organized for **selective compute**, not a broad search reset. The reconciliation/final docs keep a single belief stack—Mixture-SIB as amortized belief, CT-SMC as search-grade posterior—treat AFBS as a specialist/hard-state module, and target a fast/slow runtime split where ExIt + AFBS + pondering fire only on hard positions instead of becoming the default path. The prior-answer archive reinforces the same supervision-first, selective-search posture. ([GitHub][1])

Hydra also already exposes most of the **cheap routing signals** a budget router would need. `SearchContext` in `bridge.rs` carries Mixture-SIB, CT-SMC, AFBS, and robust-opponent context; `build_search_features` already emits mixture entropy/ESS, CT-SMC-backed Hand-EV, AFBS root delta-Q summaries, and opponent risk/stress features. AFBS already has `GameStateSnapshot`, `PonderTask`, `PonderResult`, visit counts, exit-policy extraction, and ponder/cache machinery; CT-SMC exposes ESS/resampling signals; the endgame solver already has `should_activate` and urgency logic; and the game engine provides a deterministic `BatchSimulator`, which is exactly what you want for fixed-state budget-ladder benchmarks. ([GitHub][2])

On the training side, the model/loss/data stack is already rich enough for **worst-group auxiliary robustness**. `HydraOutput` includes tenpai, danger, opponent-next-discard, belief fields, mixture weights, delta-Q, and safety-residual surfaces; `HydraLossConfig` already has weights for these heads, though the more advanced ones default to zero; and `MjaiSample` / `mjai_loader` already emit placement, score delta, GRP label, tenpai, danger, opponent-next, safety residual, and some belief/mixture targets. Two caveats matter: `grp_label` is already the 24-way GRP target, so robustness slices need a **new** field, and `to_hydra_targets()` still leaves `opponent_hand_type_target` and `delta_q_target` unset, so those should not be phase-1 robust targets. Your pasted `rl.rs` / `exit.rs` snippets also show that ExIt targets, exit weighting, and the safety valve already exist, so proposal A should upgrade the gate, not invent a new distillation path. ([GitHub][3])

2. strongest outside ingredients found

For compute routing, the strongest ingredients were: **value-of-computation** labels from metareasoning for MCTS; **learned computation selection** (BMPS / “learning to select computations”) for choosing which computation to run and when to stop; **bandit-style adaptive Monte Carlo allocation** when candidate estimators have different costs; and **budget-calibrated early-exit routing** for enforcing a fixed compute budget without spending full compute on easy instances. Those ideas map unusually well to Hydra because Hydra already has several compute arms with materially different cost profiles and trust surfaces. ([Proceedings of Machine Learning Research][4])

For robustness, the strongest package was: **group DRO** as the core objective, plus the crucial second-order lesson that naive group DRO can fail in overparameterized models unless you pair it with stronger regularization or early stopping; **JTT** as a fallback when explicit group labels are weak; and **WILDS-style evaluation discipline**, where worst-group and shifted-pool performance are first-class metrics rather than afterthoughts. ([ICLR][5])

3. final proposal A — compute routing or stronger replacement

**problem solved.**
Hydra already has multiple expensive compute arms—fast policy/value, CT-SMC/Hand-EV enrichment, AFBS, endgame solving, and ponder reuse—but the current selective-compute logic is still largely heuristic hard-state gating. That is exactly the metareasoning setting where “which computation should I run now?” should be optimized directly under a latency budget, rather than approximated by a single threshold. ([GitHub][1])

**borrowed ingredients.**
Use VOC-style labels for “expected value of extra compute,” BMPS-style learned selection over computations, adaptive-Monte-Carlo cost-aware allocation, and budget-calibrated routing ideas from early-exit work. ([Proceedings of Machine Learning Research][4])

**exact mechanism.**
Add a small stand-alone router `f(x, budget_ms) -> arm`, where `arm ∈ {ponder_hit, fast, hand_ev, afbs_small, afbs_large, endgame}`. `ponder_hit` is the lowest-cost arm: use a cached `PonderResult` if the info-state hash matches and trust checks pass. `hand_ev` means use the existing cheap search-side features without live tree expansion. `afbs_small` / `afbs_large` are measured visit/time budgets from the real engine, not fixed magic numbers. `endgame` is only eligible when `should_activate()` says the state is late and threatening. The router does **not** change ExIt target generation or safety-valve logic from your pasted `exit.rs`; it only decides whether and how much expensive computation to buy. ([GitHub][2])

The feature vector `x` should be built only from already-cheap signals: base top-2 policy gap, mixture entropy/ESS, CT-SMC ESS ratio / particle count, AFBS snapshot `risk_score` / `particle_ess`, opponent risk/stress, endgame urgency, and cached-ponder availability. [Inference] If `wall_remaining` and `legal_count` are not already passed through the runtime path you call before search, expose them on `SearchContext` rather than adding a new model head. Offline labels come from a budget ladder: for each fixed state, run a small arm ladder and assign each arm
`gain(a) = Q_ref(a) - Q_ref(fast) - λ * cost_ms(a)`,
where `Q_ref(a)` is the deepest available reference value for the action that arm would choose. Train the router to pick the best eligible arm under the current budget. Then reuse the same router score to replace or augment `PonderManager` priority, so idle-time pondering chases high-VOC states instead of just heuristic urgency. ([GitHub][2])

**Hydra insertion points.**

* `hydra-core/src/bridge.rs`: add `RoutingFeatures` plus `build_routing_features(...)` next to `build_search_features(...)`; source from `SearchContext`, Mixture-SIB, CT-SMC, AFBS root summary, and robust-opponent signals. ([GitHub][2])
* `hydra-core/src/afbs.rs`: add a `RoutingArm`/budget enum, emit measured arm outcomes for ladder logging, consume cached `PonderResult` as an arm, and route `PonderManager.enqueue_snapshot` through the learned priority score. ([GitHub][6])
* `hydra-core/src/endgame.rs`: use `should_activate` / urgency as eligibility + routing features. ([GitHub][7])
* `hydra-core/src/ct_smc.rs`: consume `ess_ratio()` / resample state as routing features only; no algorithm rewrite. ([GitHub][8])
* training side: add a tiny `router.rs` / `benchmark_gates.rs` module rather than a new Hydra head; the `BenchmarkGates` phase in your pasted `bc.rs` is the natural place to fit/evaluate it. Keep the existing `exit_target` / `exit_loss` path from your pasted `rl.rs` / `exit.rs` unchanged.

**needed labels / logs / supervision.**
No new human labels. Add a `RouteLogRecord` with cheap features, chosen arm, elapsed ms, visit count, cached-hit flag, chosen action, reference-Q label, and whether the ExIt safety valve accepted the target. Generate training data only on a curated hard-state suite at first, using `BatchSimulator` so the ladder is deterministic and re-runnable. ([GitHub][9])

**pseudocode or algorithm loop.**

```text
x = build_routing_features(state, search_ctx, base_policy)

if ponder_cache.has_trusted_hit(info_state_hash):
    return use_ponder_hit()

eligible = arms_allowed_by_budget_and_state(x, budget_ms)
arm = router.choose(x, eligible, budget_ms)

match arm:
  FAST      => act(base_policy)
  HAND_EV   => act(base_policy_with_existing_hand_ev_context)
  AFBS_K    => run_afbs(K); maybe_emit_exit_target_if_safety_valve_passes()
  ENDGAME   => run_endgame_if_eligible(); maybe_emit_exit_target_if_safety_valve_passes()

log_route_example(x, arm, elapsed_ms, chosen_action, ref_q)

# offline fitting
for state in fixed_state_hard_suite:
    outcomes = run_budget_ladder(state)
    y = argmax_a [Q_ref(a) - Q_ref(fast) - lambda * cost_ms(a)]
fit tiny router on (x, y)
```

This loop reuses existing AFBS / ponder / ExIt surfaces; it does not require a new main model head. ([GitHub][2])

**benchmark and kill criteria.**
Offline: equal-state budget ladder benchmark, measured on riichi-defense states, low-ESS states, last-10-wall states, and close-policy-gap states; report gain-per-ms and arm-choice regret against the current heuristic gate. Online: duplicate arena evaluation at equal **average** latency against the existing hard-state threshold gate. Kill it if fixed-latency duplicate results are flat/negative even when the offline surrogate improves, or if the learned policy collapses to a near-single-arm policy. The promotion gate is match results at fixed compute, not router accuracy. ([GitHub][10])

I sanity-checked the interface in Python: [hydra_prototypes.py](sandbox:/mnt/data/hydra_prototypes.py), [prototype test log](sandbox:/mnt/data/hydra_prototypes_test_output.txt), [synthetic router eval](sandbox:/mnt/data/hydra_router_synth_eval.py), [router eval log](sandbox:/mnt/data/hydra_router_synth_eval.txt). The toy ladder shows the routing objective is coherent; it is **not** evidence about Hydra strength.

4. final proposal B — worst-group robustness or stronger replacement

**problem solved.**
Hydra’s likely failure mode is not average underfitting; it is brittle behavior on high-leverage slices where opponent pressure, placement pressure, and belief misspecification interact. Average loss will underweight exactly those slices. Group-robust reweighting fits Hydra because the repo already has several stable auxiliary supervision surfaces that are closer to the real failure mode than the main RL term is. ([GitHub][1])

**borrowed ingredients.**
Use group DRO as the core objective, pair it with stronger regularization / early stopping as the original work recommends, use JTT only as a fallback when explicit groups are weak, and evaluate with WILDS-style worst-group / shifted-pool reporting rather than average-only reporting. ([ICLR][5])

**exact mechanism.**
Do **not** robustify Hydra’s whole RL objective first. In v1, leave the ACH/DRDA policy-gradient core alone and robustify only the existing supervised auxiliary surfaces that already have stable labels: tenpai, danger, opponent-next-discard, and safety-residual. Leave `delta_q` and `opponent_hand_type` out of v1 because `to_hydra_targets()` currently leaves them unset. ([GitHub][11])

Add `robust_group_id` to samples/batches. Start with **scenario slices** Hydra can derive now: placement bucket, score-delta bucket, phase bucket, and threat bucket. [Inference] If you later confirm enough coverage, add a coarse style bucket from a cheap per-game prepass over the same MJAI event stream; do not make style mandatory in v1 because sparse groups will make the optimizer noisy. The training objective should be
`L_total = L_base + β * Σ_g q_g * L_aux,g`,
where `L_base` is the untouched base loss (policy/value/score/GRP) and `L_aux,g` is the mean selected auxiliary loss for group `g`; update `q_g` with exponentiated-gradient or EMA-smoothed log-weights over group losses. If explicit groups plateau, run a JTT second stage that upweights the top-loss auxiliary examples found by a first-pass model. ([GitHub][12])

**Hydra insertion points.**

* `hydra-train/src/data/sample.rs`: add `robust_group_id: u16` (or a compact tuple) to `MjaiSample` / `MjaiBatch`. Keep `grp_label` untouched because it already feeds the 24-way GRP target. ([GitHub][12])
* `hydra-train/src/data/mjai_loader.rs`: derive `robust_group_id` from already-available placement / score / safety / game-state context; later optionally add a cheap per-game style summary pass. ([GitHub][13])
* `hydra-train/src/training/losses.rs`: add `aux_per_sample_breakdown()` plus `group_dro_reduce()`; reuse existing per-sample helpers rather than rewriting head logic. ([GitHub][11])
* your pasted `bc.rs`: tune by worst-group validation metrics, using the existing `weight_decay` and warmup/cosine schedule rather than inventing a new trainer.
* your pasted `rl.rs`: apply group weights only to the auxiliary term in v1, not to `ach_policy_loss`.
* `hydra-train/src/model.rs`: no architecture change in v1. ([GitHub][3])

**needed labels / logs / supervision.**
No new annotation source. You need only derived metadata (`robust_group_id`) and per-sample auxiliary-loss logging. Add per-group validation tables for tenpai calibration, danger error, opponent-next error, and safety-residual error; if you run JTT, cache the high-loss example ids from the first-pass model. ([GitHub][13])

**pseudocode or algorithm loop.**

```text
# loader
robust_group_id = hash(placement_bin, score_bin, phase_bin, threat_bin[, style_bin])

# train step
out = model.forward(obs)

aux_i =
    w_tenpai * loss_tenpai_i +
    w_danger * loss_danger_i +
    w_opp    * loss_opp_next_i +
    w_safe   * loss_safety_residual_i

for each active group g:
    L_aux,g = mean(aux_i[group_id == g])

log_q[g] <- log_q[g] + eta * clamp(L_aux,g, 0, Lmax)
q = softmax(log_q over active groups)

L_total = L_base(policy, value, grp, score) + beta * sum_g q[g] * L_aux,g
update(model)

# fallback if explicit groups are weak
JTT: train ERM once, mine highest-loss aux examples, upweight them in stage 2
```

This is a reduction-level change, not a new network. ([GitHub][11])

**benchmark and kill criteria.**
Offline: worst-group tenpai Brier/ECE, danger PR-AUC/ECE, opponent-next top-k, and safety-residual MSE on held-out slices. Online: duplicate arena against both balanced pools and style-biased / shifted pools, reporting average and worst-group placement EV separately. Kill it if worst-group metrics improve but balanced duplicate results go flat/negative, or if the group scheme is too sparse/noisy to stabilize `q_g`. Use WILDS-style reporting discipline: the point is not just a better average. ([Microsoft][14])

I also sanity-checked the reduction mechanics in Python: [hydra_prototypes.py](sandbox:/mnt/data/hydra_prototypes.py), [prototype test log](sandbox:/mnt/data/hydra_prototypes_test_output.txt), [synthetic group-DRO eval](sandbox:/mnt/data/hydra_group_dro_synth_eval.py), [group-DRO log](sandbox:/mnt/data/hydra_group_dro_synth_eval.txt). That toy check only verifies the optimizer behavior, not Mahjong performance.

5. what you explicitly rejected and why

* **Broader graph-search rewrites now.** Monte-Carlo Graph Search is structurally relevant, but Hydra’s current design docs explicitly keep AFBS selective/specialist and prioritize loop-closure on the existing belief/search stack first. A graph-search rewrite is too far from the repo’s current posture for the first move. ([Proceedings of Machine Learning Research][15])
* **Group-DRO on the policy-gradient core first.** Inference: Hydra’s present supervision is much cleaner on auxiliary heads than on on-policy advantages, and even the original group-DRO work says worst-group gains require careful regularization/validation in the easier supervised regime. Robustify auxiliaries first; do not start by distorting `ach_policy_loss`. ([GitHub][16])
* **Conformal risk control as the first project.** It is attractive for later calibrated danger / abstention control, but it is a post-hoc risk layer, not the highest-leverage fix for Hydra’s current compute-allocation and shift-robustness bottlenecks. ([MIT CSAIL][17])
* **A new belief-stack rewrite.** Reconciliation already chooses one Mixture-SIB + CT-SMC stack and explicitly tries to avoid duplicate machinery now. ([GitHub][1])

6. final recommendation: what Hydra should actually try first

**Try first:** proposal A, as a **stand-alone budget router** that replaces the current heuristic hard-state trigger but keeps the existing ExIt safety valve and target plumbing intact. It is the highest-upside, lowest-intrusion move because Hydra already has the compute arms, cheap routing features, ponder cache, and distillation path. Then add proposal B on auxiliary heads only, with scenario-slice groups first and style-biased evaluation immediately. ([GitHub][1])

**Best cheap benchmark first:** deterministic hard-state budget ladders from `BatchSimulator`, then equal-budget duplicate arena against the current heuristic gate. **Why this beats “bigger model” or “more search”:** A increases value per millisecond on the exact selective-search stack Hydra already wants, and B reallocates gradient to rare slices average loss underweights without an architecture reset. **Biggest hidden risk:** the router can overfit deep-search surrogate gain instead of real match EV, so fixed-latency duplicate evaluation must stay the promotion gate. ([GitHub][9])

[1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md"
[2]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs"
[3]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/model.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/model.rs"
[4]: https://proceedings.mlr.press/v124/sezener20a.html "https://proceedings.mlr.press/v124/sezener20a.html"
[5]: https://iclr.cc/virtual/2020/poster/1491 "https://iclr.cc/virtual/2020/poster/1491"
[6]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/afbs.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/afbs.rs"
[7]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/endgame.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/endgame.rs"
[8]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/ct_smc.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/ct_smc.rs"
[9]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md"
[10]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md"
[11]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/training/losses.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/training/losses.rs"
[12]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/sample.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/sample.rs"
[13]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/mjai_loader.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/mjai_loader.rs"
[14]: https://www.microsoft.com/en-us/research/publication/wilds-a-benchmark-of-in-the-wild-distribution-shifts/ "https://www.microsoft.com/en-us/research/publication/wilds-a-benchmark-of-in-the-wild-distribution-shifts/"
[15]: https://proceedings.mlr.press/v129/leurent20a.html "https://proceedings.mlr.press/v129/leurent20a.html"
[16]: https://github.com/kohpangwei/group_DRO "https://github.com/kohpangwei/group_DRO"
[17]: https://people.csail.mit.edu/tals/publication/conformal_risk/ "https://people.csail.mit.edu/tals/publication/conformal_risk/"
