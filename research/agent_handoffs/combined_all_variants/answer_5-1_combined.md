<combined_run_record run_id="005_compute_router_and_robustness" variant_id="root_followup_answer" schema_version="1">
<metadata>
<notes>Root follow-up prompt + root follow-up answer.</notes>
<layout>single_markdown_file_prompt_then_answer</layout>
</metadata>

<prompt_section>
<prompt_text status="preserved" source_path="PROMPT_5_FOLLOWUP_COMPUTE_ROUTER_AND_ROBUSTNESS.md">
<![CDATA[# Hydra follow-up for agent 5 — compute routing and worst-group robustness

No-zip, browse-tool-first version.

Assume browse/fetch tool can read raw GitHub markdown direct. Do **not** assume zip attachments exist.

Primary project-read path = raw GitHub links below.

  ## Critical directive — how to read the core Hydra docs

Avoid known bad behavior: fragmented keyword-peeking over big architecture docs.

Bad for this task:
  - keyword search first
  - isolated 20-100 line reads around keywords
  - treating docs like logs/grep DB
  - building recs from scattered snippets, not whole-doc understanding

For this task, disqualifying.

Required reading workflow:
  1. Use **browse/fetch tool on raw GitHub links** for core docs below.
  2. Read core docs **holistically + sequentially** before narrow search.
  3. Build high-level model of Hydra modules + priorities fit.
  4. Only then use narrow search for detailed impl questions.

Do **not** use grep-style keyword hunting as primary strategy for these core docs.

Core docs that must be read holistically first:
  - `research/design/HYDRA_RECONCILIATION.md`
  - `research/design/HYDRA_FINAL.md`
  - `docs/GAME_ENGINE.md`
  - `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md`
  - `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md`
  - `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md`

Only after core docs ingested holistically may you narrow to:
  - code insertion points
  - supporting design docs
  - exact file/function refs
  - outside papers + adjacent fields

<holistic_ingestion_rules>
  - Read core docs as whole docs before narrowing.
  - Do not start with keyword search on core docs.
  - Do not rely on fragmented line-window retrieval for architecture understanding.
  - After holistic reading, targeted search allowed for exact details.
</holistic_ingestion_rules>

  ## Reading order — use browse/fetch in this exact sequence

Holistic core-doc pass:
  1. `research/design/HYDRA_RECONCILIATION.md`
  2. `research/design/HYDRA_FINAL.md`
  3. `docs/GAME_ENGINE.md`
  4. `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md`
  5. `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md`
  6. `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md`

Then targeted impl grounding:
  7. `hydra-core/src/afbs.rs`
  8. `hydra-core/src/bridge.rs`
  9. `hydra-core/src/ct_smc.rs`
  10. `hydra-core/src/endgame.rs`
  11. `hydra-core/src/hand_ev.rs`
  12. `hydra-core/src/robust_opponent.rs`
  13. `hydra-train/src/data/sample.rs`
  14. `hydra-train/src/data/mjai_loader.rs`
  15. `hydra-train/src/training/losses.rs`
  16. `hydra-train/src/model.rs`

Only after that branch into outside papers + GitHub examples.

  ## Raw GitHub fallback links

Core Hydra docs:
  - `README.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/README.md
  - `research/design/HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
  - `research/design/HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
  - `research/design/HYDRA_ARCHIVE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_ARCHIVE.md
  - `research/design/IMPLEMENTATION_ROADMAP.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/IMPLEMENTATION_ROADMAP.md
  - `research/design/OPPONENT_MODELING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md
  - `research/design/TESTING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md
  - `research/design/SEEDING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/SEEDING.md
  - `docs/GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md
  - `research/infrastructure/INFRASTRUCTURE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/infrastructure/INFRASTRUCTURE.md

Thin source slices:
  - `hydra-core/src/afbs.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/afbs.rs
  - `hydra-core/src/bridge.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs
  - `hydra-core/src/ct_smc.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/ct_smc.rs
  - `hydra-core/src/endgame.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/endgame.rs
  - `hydra-core/src/hand_ev.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/hand_ev.rs
  - `hydra-core/src/robust_opponent.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/robust_opponent.rs
  - `hydra-train/src/data/sample.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/sample.rs
  - `hydra-train/src/data/mjai_loader.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/mjai_loader.rs
  - `hydra-train/src/training/losses.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/training/losses.rs
  - `hydra-train/src/model.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/model.rs

Prior answer anchors:
  - `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md
  - `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md
  - `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md

Reference links already surfaced relevant:
  - Static and Dynamic Values of Computation in MCTS — https://proceedings.mlr.press/v124/sezener20a.html
  - Distributionally Robust Neural Networks / group shift — https://iclr.cc/virtual/2020/poster/1491
  - Conformal Risk Control — https://people.csail.mit.edu/tals/publication/conformal_risk/
  - Monte-Carlo Graph Search — https://proceedings.mlr.press/v129/leurent20a.html
  - Rao-Blackwellized Particle Filter review — https://asp-eurasipjournals.springeropen.com/articles/10.1155/2010/724087

Prior memo useful but too vague cross-field transfer. This time: no broad survey.

You are GPT-5.4 Pro in long-horizon research+engineering mode. Prefer disciplined retrieval, evidence-backed synthesis, strong completion over fast ideation.

Mission: convert 2 highest-value directions from memo into code-grounded engineering + experiment brief:

  1. compute routing / selective compute allocation
  2. worst-group robust training over opponent-style or scenario slices

You may search widely across papers, official docs, GitHub repos, related systems for better ingredients, objectives, interfaces, benchmark design, but final output must ground in Hydra actual code + docs.

Do not over-constrain early. Explore broad first, then converge hard.

<output_contract>
  - Return exactly requested sections, in requested order.
  - Keep final answer compact, high-signal, impl-oriented.
  - Do not pad with generic background or repeat prompt.
</output_contract>

<verbosity_controls>
  - Prefer concise, dense writing.
  - Keep framing brief.
  - Do not over-shorten such that formulas, interfaces, benchmarks, failure checks become vague.
</verbosity_controls>

<research_mode>
  - Work in 3 passes:
    1. Ingest: read core Hydra docs holistically first via raw links or equivalent full-doc browsing.
    2. Retrieve: identify 3-6 sub-questions, search each, follow 1-2 strong second-order leads.
    3. Synthesize: resolve contradictions, reject weak transfers, write final answer.
  - Stop only when more search unlikely change final rec.
</research_mode>

<tool_persistence_rules>
  - Use browsing/search/retrieval aggressively when it materially improves correctness or novelty.
  - Prefer full-doc browse/fetch for core docs over fragmented terminal chunk reading.
  - Do not stop at first plausible transfer or first supporting paper.
  - If source seems promising, follow 1-2 second-order leads before finalizing.
  - If search path weak, abandon explicitly, move on.
</tool_persistence_rules>

<dependency_checks>
  - Before proposing impl, verify Hydra already has or can cheaply expose needed signals, labels, insertion points.
  - Do not assume data source, API surface, or training label exists until verified from raw links or retrieved evidence.
</dependency_checks>

<completeness_contract>
  - Task incomplete until every requested deliverable covered or marked [blocked].
  - If one proposal underspecified, either finish spec or downgrade/reject.
  - Do not keep proposal because interesting.
</completeness_contract>

<empty_result_recovery>
  - If search line narrow/empty, try alternate wording, adjacent fields, or stronger source before concluding no evidence.
</empty_result_recovery>

<citation_rules>
  - Cite only sources retrieved in this workflow or from raw links below.
  - Never fabricate refs.
  - Attach citations to exact claims they support.
</citation_rules>

<grounding_rules>
  - Base repo claims only on raw links below or other retrieved evidence.
  - If statement inference, label it inference.
  - If sources conflict, state conflict + resolve explicitly.
</grounding_rules>

<verification_loop>
  - Before finalizing, check:
    - did you read core Hydra docs holistically before narrowing?
    - did you explore more than one outside line of attack?
    - did you reject at least one plausible but weak transfer?
    - are file-level insertion points + required data/labels explicit?
    - is each proposal specific enough for coding agent to start?
</verification_loop>

<dig_deeper_nudge>
  - Do not stop at first plausible answer.
  - Look for second-order issues, hidden deps, edge cases, weak assumptions.
  - If answer still reads like ranking memo, not done.
</dig_deeper_nudge>

  ## What counts as success

Do not stop until all below done:

  1. exact compute-router proposal implementable in Hydra without guessing core interfaces
  2. exact worst-group/minimax-robust training proposal implementable atop current Hydra heads/data
  3. file-level insertion points in real repo
  4. algorithm sketches/pseudocode detailed enough for coding agent to start
  5. concrete benchmark + kill-criteria protocol for each proposal
  6. section explaining why these beat making Hydra bigger/more complex

  ## Strong guidance

  - Search outside Mahjong. Good fields: metareasoning, test-time compute allocation, algorithm portfolios, selective prediction, robust optimization, multi-source minimax, calibration under shift, risk-sensitive decision systems.
  - You may find different outside idea stronger than original compute-router / worst-group pair. If so, say so clearly, but only if defensible to higher standard.
  - You may synthesize new variants/hybrids if papers alone insufficient.
  - Keep final rec small. One or two serious routes beat laundry list.

  ## Hydra-specific grounding requirements

Ground every serious proposal in:
  - exact files/functions/structs likely to change
  - what data already exists + what new logging/labels needed
  - what can ship now vs what needs later infra
  - how it interacts with current AFBS / Hand-EV / belief / search-trust / danger / tenpai surfaces

  ## Required output format

  1. what Hydra already has that matters for this question
  2. strongest outside ingredients found
  3. final proposal — compute routing or stronger replacement
     - problem solved
     - borrowed ingredients
     - exact mechanism
     - Hydra insertion points
     - needed labels / logs / supervision
     - pseudocode or algorithm loop
     - benchmark and kill criteria
  4. final proposal B — worst-group robustness or stronger replacement
     - same sub-sections as above
  5. what you explicitly rejected and why
  6. final rec: what Hydra should try first

  ## Hard constraints

  - no generic "use transformers", "use more search", or do bigger RL"
  - no AGPL-derived code advice
  - no shallow ranking memo
  - no pretending proposal code-ready if key interfaces or labels still unknown

Success = output specific enough that coding agent can translate into real work plan with minimal guesswork.
]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="preserved" source_path="agent_5-followup.md">
<![CDATA[1. what Hydra already has that matters for this question

Hydra already organized for **selective compute**, not broad search reset. Reconciliation/final docs keep single belief stack—Mixture-SIB as amortized belief, CT-SMC as search-grade posterior—treat AFBS as specialist/hard-state module, target fast/slow runtime split where ExIt + AFBS + pondering fire only on hard spots instead of default path. Prior-answer archive says same: supervision-first, selective-search posture. ([GitHub][1])

Hydra also already exposes most **cheap routing signals** budget router needs. `SearchContext` in `bridge.rs` carries Mixture-SIB, CT-SMC, AFBS, robust-opponent context; `build_search_features` already emits mixture entropy/ESS, CT-SMC-backed Hand-EV, AFBS root delta-Q summaries, opponent risk/stress features. AFBS already has `GameStateSnapshot`, `PonderTask`, `PonderResult`, visit counts, exit-policy extraction, ponder/cache machinery; CT-SMC exposes ESS/resampling; endgame solver already has `should_activate` + urgency logic; game engine provides deterministic `BatchSimulator`, ideal for fixed-state budget-ladder benchmarks. ([GitHub][2])

On training side, model/loss/data stack already rich enough for **worst-group auxiliary robustness**. `HydraOutput` includes tenpai, danger, opponent-next-discard, belief fields, mixture weights, delta-Q, safety-residual surfaces; `HydraLossConfig` already has weights for these heads, though advanced ones default zero; `MjaiSample` / `mjai_loader` already emit placement, score delta, GRP label, tenpai, danger, opponent-next, safety residual, some belief/mixture targets. Two caveats matter: `grp_label` already serves 24-way GRP target, so robustness slices need **new** field, and `to_hydra_targets()` still leaves `opponent_hand_type_target` + `delta_q_target` unset, so do not use those as phase-1 robust targets. Pasted `rl.rs` / `exit.rs` snippets also show ExIt targets, exit weighting, safety valve already exist, so proposal should upgrade gate, not invent new distillation path. ([GitHub][3])

  2. strongest outside ingredients found

For compute routing, strongest ingredients: **value-of-computation** labels from metareasoning for MCTS; **learned computation selection** (BMPS / “learning to select computations”) for choosing computation + stop point; **bandit-style adaptive Monte Carlo allocation** when estimators have different costs; **budget-calibrated early-exit routing** for enforcing fixed compute budget without full-compute spend on easy cases. These map well to Hydra because Hydra already has several compute arms with materially different cost profiles + trust surfaces. ([Proceedings of Machine Learning Research][4])

For robustness, strongest package: **group DRO** as core objective, plus key second-order lesson that naive group DRO can fail in overparameterized models unless paired with stronger regularization or early stopping; **JTT** as fallback when explicit group labels weak; **WILDS-style evaluation discipline**, where worst-group + shifted-pool performance are first-class metrics, not afterthoughts. ([ICLR][5])

  3. final proposal — compute routing or stronger replacement

**problem solved.**
Hydra already has multiple expensive compute arms—fast policy/value, CT-SMC/Hand-EV enrichment, AFBS, endgame solving, ponder reuse—but current selective-compute logic still mostly heuristic hard-state gating. This is exact metareasoning setting where “which computation should run now?” should be optimized direct under latency budget, not approximated by one threshold. ([GitHub][1])

**borrowed ingredients.**
Use VOC-style labels for “expected value of extra compute,” BMPS-style learned selection over computations, adaptive-Monte-Carlo cost-aware allocation, budget-calibrated routing ideas from early-exit work. ([Proceedings of Machine Learning Research][4])

**exact mechanism.**
Add small stand-alone router `f(x, budget_ms) -> arm`, where `arm ∈ {ponder_hit, fast, hand_ev, afbs_small, afbs_large, endgame}`. `ponder_hit` = lowest-cost arm: use cached `PonderResult` if info-state hash matches + trust checks pass. `hand_ev` = use existing cheap search-side features without live tree expansion. `afbs_small` / `afbs_large` = measured visit/time budgets from real engine, not magic constants. `endgame` eligible only when `should_activate()` says late + threatening state. Router does **not** change ExIt target generation or safety-valve logic from pasted `exit.rs`; it only decides whether/how much expensive compute to buy. ([GitHub][2])

Feature vector `x` should use only already-cheap signals: base top-2 policy gap, mixture entropy/ESS, CT-SMC ESS ratio / particle count, AFBS snapshot `risk_score` / `particle_ess`, opponent risk/stress, endgame urgency, cached-ponder availability. [Inference] If `wall_remaining` + `legal_count` not already passed through runtime path before search, expose them on `SearchContext` rather than adding new model head. Offline labels come from budget ladder: for each fixed state, run small arm ladder and assign each arm
`gain(a) = Q_ref(a) - Q_ref(fast) - λ * cost_ms(a)`,
where `Q_ref(a)` = deepest available reference value for action that arm would choose. Train router to pick best eligible arm under current budget. Then reuse same router score to replace/augment `PonderManager` priority, so idle-time pondering chases high-VOC states instead of heuristic urgency only. ([GitHub][2])

**Hydra insertion points.**

  * `hydra-core/src/bridge.rs`: add `RoutingFeatures` + `build_routing_features(...)` next to `build_search_features(...)`; source from `SearchContext`, Mixture-SIB, CT-SMC, AFBS root summary, robust-opponent signals. ([GitHub][2])
  * `hydra-core/src/afbs.rs`: add `RoutingArm`/budget enum, emit measured arm outcomes for ladder logging, consume cached `PonderResult` as arm, route `PonderManager.enqueue_snapshot` through learned priority score. ([GitHub][6])
  * `hydra-core/src/endgame.rs`: use `should_activate` / urgency as eligibility + routing features. ([GitHub][7])
  * `hydra-core/src/ct_smc.rs`: consume `ess_ratio()` / resample state as routing features only; no algorithm rewrite. ([GitHub][8])
  * training side: add tiny `router.rs` / `benchmark_gates.rs` module, not new Hydra head; pasted `bc.rs` `BenchmarkGates` phase is natural fit/eval place. Keep existing `exit_target` / `exit_loss` path from pasted `rl.rs` / `exit.rs` unchanged.

**needed labels / logs / supervision.**
No new human labels. Add `RouteLogRecord` with cheap features, chosen arm, elapsed ms, visit count, cached-hit flag, chosen action, reference-Q label, whether ExIt safety valve accepted target. Generate training data only on curated hard-state suite first, using `BatchSimulator` so ladder deterministic + rerunnable. ([GitHub][9])

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

This loop reuses existing AFBS / ponder / ExIt surfaces; no new main model head needed. ([GitHub][2])

**benchmark and kill criteria.**
Offline: equal-state budget ladder benchmark on riichi-defense states, low-ESS states, last-10-wall states, close-policy-gap states; report gain-per-ms + arm-choice regret against current heuristic gate. Online: duplicate arena eval at equal **average** latency against existing hard-state threshold gate. Kill if fixed-latency duplicate results flat/negative even when offline surrogate improves, or if learned policy collapses to near-single-arm policy. Promotion gate = match results at fixed compute, not router accuracy. ([GitHub][10])

I sanity-checked interface in Python: [hydra_prototypes.py](sandbox:/mnt/data/hydra_prototypes.py), [prototype test log](sandbox:/mnt/data/hydra_prototypes_test_output.txt), [synthetic router eval](sandbox:/mnt/data/hydra_router_synth_eval.py), [router eval log](sandbox:/mnt/data/hydra_router_synth_eval.txt). Toy ladder shows routing objective coherent; **not** evidence about Hydra strength.

  4. final proposal B — worst-group robustness or stronger replacement

**problem solved.**
Hydra likely failure mode is not average underfitting; it is brittle behavior on high-leverage slices where opponent pressure, placement pressure, belief misspecification interact. Average loss underweights exactly those slices. Group-robust reweighting fits Hydra because repo already has several stable auxiliary supervision surfaces closer to real failure mode than main RL term. ([GitHub][1])

**borrowed ingredients.**
Use group DRO as core objective, pair with stronger regularization / early stopping as original work recommends, use JTT only as fallback when explicit groups weak, evaluate with WILDS-style worst-group / shifted-pool reporting rather than average-only reporting. ([ICLR][5])

**exact mechanism.**
Do **not** robustify Hydra whole RL objective first. In v1, leave ACH/DRDA policy-gradient core alone and robustify only existing supervised auxiliary surfaces with stable labels: tenpai, danger, opponent-next-discard, safety-residual. Leave `delta_q` + `opponent_hand_type` out of v1 because `to_hydra_targets()` leaves them unset. ([GitHub][11])

Add `robust_group_id` to samples/batches. Start with **scenario slices** Hydra can derive now: placement bucket, score-delta bucket, phase bucket, threat bucket. [Inference] If later confirmed enough coverage, add coarse style bucket from cheap per-game prepass over same MJAI event stream; do not make style mandatory in v1 because sparse groups make optimizer noisy. Training objective should be
`L_total = L_base + β * Σ_g q_g * L_aux,g`,
where `L_base` = untouched base loss (policy/value/score/GRP) and `L_aux,g` = mean selected auxiliary loss for group `g`; update `q_g` with exponentiated-gradient or EMA-smoothed log-weights over group losses. If explicit groups plateau, run JTT second stage that upweights top-loss auxiliary examples found by first-pass model. ([GitHub][12])

**Hydra insertion points.**

  * `hydra-train/src/data/sample.rs`: add `robust_group_id: u16` (or compact tuple) to `MjaiSample` / `MjaiBatch`. Keep `grp_label` untouched because it already feeds 24-way GRP target. ([GitHub][12])
  * `hydra-train/src/data/mjai_loader.rs`: derive `robust_group_id` from already-available placement / score / safety / game-state context; later optionally add cheap per-game style summary pass. ([GitHub][13])
  * `hydra-train/src/training/losses.rs`: add `aux_per_sample_breakdown()` + `group_dro_reduce()`; reuse existing per-sample helpers, not rewrite head logic. ([GitHub][11])
  * pasted `bc.rs`: tune by worst-group validation metrics, using existing `weight_decay` + warmup/cosine schedule rather than invent new trainer.
  * pasted `rl.rs`: apply group weights only to auxiliary term in v1, not `ach_policy_loss`.
  * `hydra-train/src/model.rs`: no architecture change in v1. ([GitHub][3])

**needed labels / logs / supervision.**
No new annotation source. Need only derived metadata (`robust_group_id`) + per-sample auxiliary-loss logging. Add per-group validation tables for tenpai calibration, danger error, opponent-next error, safety-residual error; if running JTT, cache high-loss example ids from first-pass model. ([GitHub][13])

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

This is reduction-level change, not new network. ([GitHub][11])

**benchmark and kill criteria.**
Offline: worst-group tenpai Brier/ECE, danger PR-AUC/ECE, opponent-next top-k, safety-residual MSE on held-out slices. Online: duplicate arena against both balanced pools and style-biased / shifted pools, reporting average + worst-group placement EV separately. Kill if worst-group metrics improve but balanced duplicate results go flat/negative, or if group scheme too sparse/noisy to stabilize `q_g`. Use WILDS-style reporting discipline: point is not better average. ([Microsoft][14])

I also sanity-checked reduction mechanics in Python: [hydra_prototypes.py](sandbox:/mnt/data/hydra_prototypes.py), [prototype test log](sandbox:/mnt/data/hydra_prototypes_test_output.txt), [synthetic group-DRO eval](sandbox:/mnt/data/hydra_group_dro_synth_eval.py), [group-DRO log](sandbox:/mnt/data/hydra_group_dro_synth_eval.txt). Toy check verifies optimizer behavior only, not Mahjong performance.

  5. what you explicitly rejected and why

  * **Broader graph-search rewrites now.** Monte-Carlo Graph Search structurally relevant, but Hydra current design docs explicitly keep AFBS selective/specialist and prioritize loop-closure on existing belief/search stack first. Graph-search rewrite too far from repo current posture for first move. ([Proceedings of Machine Learning Research][15])
  * **Group-DRO on policy-gradient core first.** [Inference] Hydra present supervision much cleaner on auxiliary heads than on on-policy advantages, and original group-DRO work also says worst-group gains need careful regularization/validation even in easier supervised regime. Robustify auxiliaries first; do not start by distorting `ach_policy_loss`. ([GitHub][16])
  * **Conformal risk control as first project.** Attractive later for calibrated danger / abstention control, but post-hoc risk layer, not highest-leverage fix for Hydra current compute-allocation + shift-robustness bottlenecks. ([MIT CSAIL][17])
  * **New belief-stack rewrite.** Reconciliation already chooses one Mixture-SIB + CT-SMC stack and explicitly tries to avoid duplicate machinery now. ([GitHub][1])

  6. final rec: what Hydra should try first

**Try first:** proposal as **stand-alone budget router** replacing current heuristic hard-state trigger while keeping existing ExIt safety valve + target plumbing intact. Highest-upside, lowest-intrusion move because Hydra already has compute arms, cheap routing features, ponder cache, distillation path. Then add proposal B on auxiliary heads only, with scenario-slice groups first and style-biased evaluation immediately. ([GitHub][1])

**Best cheap benchmark first:** deterministic hard-state budget ladders from `BatchSimulator`, then equal-budget duplicate arena against current heuristic gate. **Why this beats “bigger model” or “more search”:** increases value per millisecond on exact selective-search stack Hydra already wants, and B reallocates gradient to rare slices average loss underweights without architecture reset. **Biggest hidden risk:** router can overfit deep-search surrogate gain instead of real match EV, so fixed-latency duplicate evaluation must stay promotion gate. ([GitHub][9])

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
]]>
</answer_text>
</answer_section>
</combined_run_record>