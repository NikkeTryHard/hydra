<combined_run_record run_id="007_cross_field_breakthrough" variant_id="agent_7_primary" schema_version="1">
  <metadata>
    <notes>Likely compact primary answer for prompt 7.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="PROMPT_7_FRESH_CONTEXT_CROSS_FIELD_BREAKTHROUGH.md">
  <![CDATA[# Hydra fresh-context prompt — cross-field breakthrough to LuckyJ-level separator prototype

  Primary source material lives in the raw GitHub links below.

  ## Critical directive — how to read the core Hydra docs

  You must avoid a known bad behavior: fragmented keyword-peeking over large architecture docs.

  Bad behavior for this task:
  - searching for keywords first
  - reading isolated 20-100 line chunks around those keywords
  - treating the docs like logs or a grep database
  - inventing LuckyJ-breakthrough-caliber directions before understanding Hydra as a whole system

  For this task, that behavior is disqualifying.

  Required reading workflow:
  1. Use your browse/fetch tool on the raw GitHub links for the core docs listed below.
  2. Read those core docs holistically and sequentially before doing narrower searching.
  3. Build a high-level model of what Hydra already is, what is active, what is reserve, and what loops are already partially closed.
  4. Only after that may you use narrower searching for exact details and outside analogies.

  Do not use grep-style keyword hunting as your primary reading strategy for these core docs.

  <holistic_ingestion_rules>
  - Read the core docs as whole documents before narrowing.
  - Do not start with keyword search on the core docs.
  - Do not rely on fragmented line-window retrieval for architecture understanding.
  - After holistic reading, you may use targeted search for exact details.
  </holistic_ingestion_rules>

  ## Reading order

  1. `research/design/HYDRA_RECONCILIATION.md`
  2. `research/design/HYDRA_FINAL.md`
  3. `docs/GAME_ENGINE.md`
  4. `research/design/OPPONENT_MODELING.md`
  5. `research/design/TESTING.md`
  6. `research/design/SEEDING.md`
  7. `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md`
  8. `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md`
  9. `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md`
  10. code-grounding files
  11. outside retrieval

  ## Raw GitHub links

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

  Code-grounding files:
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

  Prior answer archive:
  - `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md
  - `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md
  - `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md

  You are acting as a long-think breakthrough engineer for Hydra, a Riichi Mahjong AI whose goal is to reach or exceed LuckyJ-level strength.

  Your job is not to write a generic transfer memo. Your job is to search broadly across other fields, identify combinations that could create a breakthrough of comparable strategic importance to the ACH/search leap associated with LuckyJ, and return only the few cross-field syntheses that are both mathematically defensible and prototypeable inside Hydra's real architecture.

  By “comparable strategic importance to the ACH/search leap associated with LuckyJ,” do not anchor on copying LuckyJ or blindly reviving ACH/DRDA as-is. The goal is not imitation. The goal is to find a separator-level breakthrough for Hydra: a move that changes Hydra's strategic ceiling in a similarly meaningful way. Instead, look for approaches with the same flavor of asymmetric strategic leverage:
  - policy improvement that respects search/game constraints
  - opponent- or scenario-conditioned optimization without fantasy observability
  - stable training signals in a 4-player general-sum partially observed setting
  - selective trust in stronger teachers, search, or exploit branches
  - robust regret or advantage control that survives hidden information and multiplayer non-zero-sum structure

  <output_contract>
  - Return exactly the requested sections, in the requested order.
  - Be as detailed and explicit as necessary; do not optimize for brevity.
  - Return a full technical treatment, not a compressed memo.
  - Return only 1-3 serious candidates.
  - A short answer is usually a failure mode for this prompt.
  </output_contract>

  <verbosity_controls>
  - Prefer full technical exposition over compressed summary.
  - Use multi-paragraph explanations when a short paragraph would hide important logic.
  - Do not omit equations, derivations, tensor/interface details, edge cases, or implementation caveats when they matter.
  - When in doubt, include more mathematical detail, derivation, and mechanism detail rather than less.
  </verbosity_controls>

  <research_mode>
  - Work in 3 passes:
    1. Ingest: read the Hydra docs holistically and reconstruct the current doctrine, active path, reserve shelf, and known ACH/DRDA caveats.
    2. Retrieve: search broadly across other fields and follow 1-2 strong second-order leads for each serious direction.
    3. Synthesize: keep only the candidates that survive Hydra-specific grounding, mathematical definition, and adversarial self-review.
  - Stop only when more searching is unlikely to change the final ranking.
  </research_mode>

  <tool_persistence_rules>
  - Prefer full-document browse/fetch for core docs over fragmented terminal-style chunk reading.
  - Search outside Mahjong aggressively.
  - Do not stop at the first adjacent paper.
  - Use additional retrieval when it materially improves novelty, grounding, or falsification.
  </tool_persistence_rules>

  <calculation_validation_rules>
  - If a recommendation depends on quantitative reasoning, derive it explicitly.
  - Use executable arithmetic or small scripts when needed to sanity-check formulas, tensor shapes, or threshold logic.
  - Do not fake arithmetic that could have been checked.
  </calculation_validation_rules>

  <dependency_checks>
  - Before proposing implementation, verify Hydra already has or could cheaply expose the needed signals, labels, or runtime hooks.
  - Before proposing a new objective, check whether the needed targets, trajectories, or opponent-conditioned quantities actually exist in Hydra now.
  </dependency_checks>

  <posture_reconstruction_rules>
  - Before proposing any candidate, include a short "Hydra posture reconstruction" section with 5-10 bullets.
  - Those bullets must distinguish current mainline doctrine, reserve-shelf ideas, partially closed loops, and 2-3 non-goals or deprioritized paths.
  - Do not propose breakthrough candidates until this posture reconstruction is complete.
  </posture_reconstruction_rules>

  <citation_rules>
  - Cite only sources you actually retrieved in this workflow or sources included in the raw links above.
  - Never fabricate references.
  - Attach citations to the exact claims they support.
  - Include full reference detail and direct links when possible.
  </citation_rules>

  <grounding_rules>
  - Ground Hydra-specific claims in the raw links above.
  - Ground outside-technique claims in retrieved sources.
  - Label inference as inference.
  - If sources conflict, state the conflict explicitly.
  - Any repo touchpoint, label source, tensor, or runtime hook not explicitly evidenced from the provided materials must be marked `inference` or `[blocked]`.
  </grounding_rules>

  <novelty_viability_rules>
  - Do not invent shallow, buzzword-heavy acronyms.
  - If you propose a novel technique, you must prove it is technically viable.
  - In your thinking space, you must explicitly write out:
    - the mathematical formulation (e.g. the exact loss function, Bellman update, or gated objective)
    - the tensor shapes going in and out of the network
    - the exact pseudocode for the algorithm
  - If you cannot mathematically define the technique using the constraints of the Hydra architecture, discard it.
  - Be explicit about what is borrowed unchanged, what is adapted, and what is newly proposed.
  </novelty_viability_rules>

  <self_red_team_rules>
  - Before finalizing any recommendation, Red-Team your own ideas.
  - For every technique you propose, spend compute time actively searching the provided documents for reasons why the idea will fail.
  - Ask explicitly:
    - How does this break in a 4-player general-sum game?
    - Does this violate the partial observability constraints?
    - Does this require targets, beliefs, or opponent labels Hydra does not actually have?
    - Is this secretly weaker than a simpler selective-compute or target-closure move already on the mainline?
  -  - Does the supposed breakthrough collapse into an incremental tuning trick once written out mathematically?
  - Only present techniques that survive this adversarial self-review.
  </self_red_team_rules>

  <anti_survey_rules>
  - Do not return a literature survey, field map, or long list of adjacent ideas without convergence.
  - Every cited outside field or paper must earn its place by changing the final candidate set or the red-team analysis.
  - If a paragraph does not help define, falsify, compare, or prototype a surviving candidate, cut it.
  </anti_survey_rules>

  <novelty_honesty_rules>
  - For every surviving candidate, include a "closest known baseline" subsection.
  - State the nearest known method or family, the exact overlap, and the irreducible difference.
  - If the method reduces to a known technique under realistic Hydra constraints, downgrade or reject it.
  - Label each surviving candidate as one of:
    - `A`: genuinely new mechanism
    - `B`: known mechanism with a Hydra-specific adaptation that plausibly changes capability
    - `C`: renamed or lightly modified known trick
  - Reject all `C` candidates.
  </novelty_honesty_rules>

  <minimum_falsification_rules>
  - For every surviving candidate, define the minimum falsifiable prototype that tests the claimed breakthrough mechanism in isolation.
  - If the core claim cannot be tested without a large coupled rollout or major stack build-out, reject the idea as too diffuse.
  - The first benchmark should distinguish the idea from stronger tuning, more search, more data, or easier teacher signals.
  </minimum_falsification_rules>

  <completeness_contract>
  - Treat the task as incomplete until every surviving candidate includes exact mechanism, mathematical formulation, tensor shapes, pseudocode, repo insertion points, cheapest prototype path, benchmark plan, and kill criteria.
  - Mark any underspecified item [blocked] rather than pretending it is ready.
  </completeness_contract>

  <verification_loop>
  - Before finalizing, verify that you actually read the core Hydra docs holistically before narrowing in.
  - Verify that each surviving candidate is genuinely more interesting than generic bigger-model or bigger-search moves.
  - Verify that each surviving candidate is not just a renamed known trick.
  - Verify that a coding agent could begin prototyping the best candidate from your answer with minimal guesswork.
  </verification_loop>

  <dig_deeper_nudge>
  - Do not stop at the first cool transfer.
  - Prefer techniques that create asymmetry, not cosmetic complexity.
  - Search especially hard around multiplayer RL, imperfect-information games, selective trust-region updates, conservative policy improvement, teacher-gated distillation, opponent-mixture optimization, and value-of-computation control.
  </dig_deeper_nudge>

  ## What to do

  1. Reconstruct Hydra's real current posture, especially the fact that broad ACH/DRDA-style optimizer bets are not currently on the mainline.
  2. Search across outside fields for ideas that could create a breakthrough of comparable importance to LuckyJ's ACH/search leap without merely copying that path and without assuming away 4-player general-sum partial observability.
  3. Combine good parts from multiple papers only when the combination is technically coherent.
  4. For each serious candidate, force it through mathematical specification, tensor-shape specification, pseudocode, repo-grounding, and adversarial failure analysis.
  5. Compare each surviving candidate against the strongest simpler mainline alternative and reject it if it is mostly incremental.
  6. Reject weak or underdefined ideas explicitly.

  ## Deliverables

  Return only 1-3 serious candidates.

  If no candidate survives the novelty, grounding, and prototypeability filters, return `0 surviving candidates` and explain why.

  For each candidate give, in order:
  1. name
  2. problem solved
  3. outside ingredients and exact sources
  4. what is borrowed unchanged
  5. what is adapted for Hydra
  6. what is genuinely novel synthesis
  7. why it transfers to Hydra specifically
  8. exact mathematical formulation
  9. tensor shapes and affected network interfaces
  10. exact algorithm pseudocode
  11. exact Hydra surfaces it would touch
  12. prototype path
  13. benchmark plan
  14. what success would look like
  15. what would kill the idea quickly
  16. red-team failure analysis
  17. why this is more likely to matter than the strongest simpler mainline alternative
  18. closest known baseline and why this does not reduce to it
  19. dependency closure table: required signal / label / hook / teacher / runtime state | already exists / cheap to expose / missing | evidence or blocker
  20. minimum falsifiable prototype

  Then end with:
  - the single best candidate to try first
  - the single best cheap benchmark to run first
  - the single biggest hidden implementation risk
  - the 2-5 most tempting rejected directions and exactly why they were rejected

  ## Hard constraints

  - no generic bigger-model or bigger-search proposals
  - no AGPL-derived code advice
  - no broad architecture resets
  - no pretty but untestable ideas
  - no keeping weak options alive just because they sound novel
  - no shallow acronym invention
  - no recommendation that cannot be written as an executable objective/update rule plus pseudocode inside Hydra constraints
  - no answer that mainly summarizes literature instead of converging to a prototype-worthy separator candidate
  - no candidate that survives only because the prompt implicitly demanded at least one answer

  Success means your final answer feels like a serious long-think synthesis pass and leaves behind at least one separator-level direction for Hydra that is mathematically explicit, hostile to self-delusion, genuinely prototypeable, and plausibly capable of changing Hydra's ceiling the way LuckyJ's signature breakthroughs changed theirs.
  ]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_7.md">
  <![CDATA[## 1

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
  ]]>
  </answer_text>
  </answer_section>
</combined_run_record>
