<combined_run_record run_id="007_cross_field_breakthrough" variant_id="agent_7_primary" schema_version="1">
<metadata>
<notes>Compact primary answer for prompt 7.</notes>
<layout>single_markdown_file_prompt_then_answer</layout>
</metadata>

<prompt_section>
<prompt_text status="preserved" source_path="PROMPT_7_FRESH_CONTEXT_CROSS_FIELD_BREAKTHROUGH.md">
<![CDATA[# Hydra fresh-context prompt — cross-field breakthrough to LuckyJ-level separator prototype

Primary source material in raw GitHub links below.

  ## Critical directive — how to read the core Hydra docs

Avoid known bad behavior: fragmented keyword-peeking over large architecture docs.

Bad behavior for task:
  - keyword search first
  - read isolated 20-100 line chunks near keywords
  - treat docs like logs or grep DB
  - invent LuckyJ-breakthrough-grade directions before understanding Hydra whole-system

For task, disqualifying.

Required reading workflow:
  1. Use browse/fetch on raw GitHub links for core docs below.
  2. Read core docs whole and sequentially before narrower search.
  3. Build high-level model of what Hydra already is, what active, what reserve, what loops already partly closed.
  4. Only then use narrower search for exact details and outside analogies.

Do not use grep-style keyword hunting as primary reading strategy for core docs.

<holistic_ingestion_rules>
  - Read core docs as whole docs before narrowing.
  - Do not start with keyword search on core docs.
  - Do not rely on fragmented line-window retrieval for architecture understanding.
  - After holistic reading, targeted search allowed for exact details.
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

Act as long-think breakthrough engineer for Hydra, Riichi Mahjong AI aiming for LuckyJ-level or better strength.

Job not generic transfer memo. Search broad across other fields, find combinations that could create breakthrough comparable in strategic importance to LuckyJ ACH/search leap, return only few cross-field syntheses both mathematically defensible and prototypeable inside real Hydra architecture.

“Comparable strategic importance to ACH/search leap” does not mean copy LuckyJ or blindly revive ACH/DRDA. Goal = separator-level breakthrough for Hydra: move that materially changes strategic ceiling. Seek approaches with same asymmetric leverage flavor:
  - policy improvement that respects search/game constraints
  - opponent- or scenario-conditioned optimization without fantasy observability
  - stable training signals in 4-player general-sum partial-observation setting
  - selective trust in stronger teachers, search, or exploit branches
  - robust regret or advantage control that survives hidden information and multiplayer non-zero-sum structure

<output_contract>
  - Return exactly requested sections, requested order.
  - Be as detailed and explicit as needed; do not optimize for brevity.
  - Return full technical treatment, not compressed memo.
  - Return only 1-3 serious candidates.
  - Short answer usually = failure mode for prompt.
</output_contract>

<verbosity_controls>
  - Prefer full technical exposition over compressed summary.
  - Use multi-paragraph explanation when short paragraph would hide important logic.
  - Do not omit equations, derivations, tensor/interface details, edge cases, or impl caveats when they matter.
  - When unsure, include more math detail, derivation, mechanism detail.
</verbosity_controls>

<research_mode>
  - Work in 3 passes:
    1. Ingest: read Hydra docs holistically, reconstruct current doctrine, active path, reserve shelf, known ACH/DRDA caveats.
    2. Retrieve: search broad across other fields, follow 1-2 strong second-order leads per serious direction.
    3. Synthesize: keep only candidates that survive Hydra-specific grounding, mathematical definition, adversarial self-review.
  - Stop only when more search unlikely to change final ranking.
</research_mode>

<tool_persistence_rules>
  - Prefer full-document browse/fetch for core docs over fragmented terminal-style chunk reading.
  - Search outside Mahjong aggressively.
  - Do not stop at first adjacent paper.
  - Use more retrieval when it materially improves novelty, grounding, or falsification.
</tool_persistence_rules>

<calculation_validation_rules>
  - If rec depends on quantitative reasoning, derive explicitly.
  - Use executable arithmetic or small scripts when needed to sanity-check formulas, tensor shapes, or threshold logic.
  - Do not fake arithmetic that could have been checked.
</calculation_validation_rules>

<dependency_checks>
  - Before proposing impl, verify Hydra already has or could cheaply expose needed signals, labels, or runtime hooks.
  - Before proposing new objective, check whether needed targets, trajectories, or opponent-conditioned quantities exist in Hydra now.
</dependency_checks>

<posture_reconstruction_rules>
  - Before proposing candidate, include short "Hydra posture reconstruction" section with 5-10 bullets.
  - Bullets must distinguish current mainline doctrine, reserve-shelf ideas, partially closed loops, and 2-3 non-goals or deprioritized paths.
  - Do not propose breakthrough candidates until posture reconstruction complete.
</posture_reconstruction_rules>

<citation_rules>
  - Cite only sources retrieved in workflow or included in raw links above.
  - Never fabricate references.
  - Attach citations to exact claims supported.
  - Include full reference detail and direct links when possible.
</citation_rules>

<grounding_rules>
  - Ground Hydra-specific claims in raw links above.
  - Ground outside-technique claims in retrieved sources.
  - Label inference as inference.
  - If sources conflict, state conflict explicitly.
  - Any repo touchpoint, label source, tensor, or runtime hook not explicitly evidenced from provided materials must be marked `inference` or `[blocked]`.
</grounding_rules>

<novelty_viability_rules>
  - Do not invent shallow buzzword acronyms.
  - If proposing novel technique, prove technical viability.
  - In thinking space, explicitly write:
    - mathematical formulation (exact loss, Bellman update, or gated objective)
    - tensor shapes in/out of network
    - exact algorithm pseudocode
  - If technique cannot be mathematically defined under Hydra constraints, discard.
  - Be explicit: what borrowed unchanged, what adapted, what newly proposed.
</novelty_viability_rules>

<self_red_team_rules>
  - Before finalizing rec, red-team own ideas.
  - For every proposed technique, spend compute time actively searching provided docs for failure reasons.
  - Ask explicitly:
    - How does this break in 4-player general-sum game?
    - Does this violate partial observability constraints?
    - Does this require targets, beliefs, or opponent labels Hydra does not have?
    - Is this secretly weaker than simpler selective-compute or target-closure move already on mainline?
    - Does supposed breakthrough collapse into incremental tuning trick once written mathematically?
  - Present only techniques surviving adversarial self-review.
</self_red_team_rules>

<anti_survey_rules>
  - Do not return literature survey, field map, or long list of adjacent ideas without convergence.
  - Every cited outside field or paper must earn place by changing final candidate set or red-team analysis.
  - If paragraph does not help define, falsify, compare, or prototype surviving candidate, cut.
</anti_survey_rules>

<novelty_honesty_rules>
  - For every surviving candidate, include "closest known baseline" subsection.
  - State nearest known method/family, exact overlap, irreducible difference.
  - If method reduces to known technique under realistic Hydra constraints, downgrade or reject.
  - Label each surviving candidate:
    - `A`: genuinely new mechanism
    - `B`: known mechanism with Hydra-specific adaptation plausibly changing capability
    - `C`: renamed or lightly modified known trick
  - Reject all `C` candidates.
</novelty_honesty_rules>

<minimum_falsification_rules>
  - For every surviving candidate, define minimum falsifiable prototype that tests claimed breakthrough mechanism in isolation.
  - If core claim cannot be tested without large coupled rollout or major stack build-out, reject as too diffuse.
  - First benchmark should distinguish idea from stronger tuning, more search, more data, or easier teacher signals.
</minimum_falsification_rules>

<completeness_contract>
  - Treat task incomplete until every surviving candidate includes exact mechanism, mathematical formulation, tensor shapes, pseudocode, repo insertion points, cheapest prototype path, benchmark plan, and kill criteria.
  - Mark underspecified item [blocked], never pretend ready.
</completeness_contract>

<verification_loop>
  - Before finalizing, verify core Hydra docs were read holistically before narrowing.
  - Verify each surviving candidate genuinely more interesting than generic bigger-model or bigger-search moves.
  - Verify each surviving candidate not merely renamed known trick.
  - Verify coding agent could start prototyping best candidate from answer with minimal guesswork.
</verification_loop>

<dig_deeper_nudge>
  - Do not stop at first cool transfer.
  - Prefer techniques that create asymmetry, not cosmetic complexity.
  - Search especially hard around multiplayer RL, imperfect-information games, selective trust-region updates, conservative policy improvement, teacher-gated distillation, opponent-mixture optimization, and value-of-computation control.
</dig_deeper_nudge>

  ## What to do

  1. Reconstruct Hydra's real current posture, especially that broad ACH/DRDA-style optimizer bets are not current mainline.
  2. Search outside fields for ideas that could create breakthrough comparable in importance to LuckyJ's ACH/search leap without merely copying that path and without assuming away 4-player general-sum partial observability.
  3. Combine good parts from multiple papers only when combination technically coherent.
  4. For each serious candidate, force through mathematical specification, tensor-shape specification, pseudocode, repo-grounding, and adversarial failure analysis.
  5. Compare each surviving candidate against strongest simpler mainline alternative; reject if mostly incremental.
  6. Reject weak or underdefined ideas explicitly.

  ## Deliverables

Return only 1-3 serious candidates.

If no candidate survives novelty, grounding, and prototypeability filters, return `0 surviving candidates` and explain why.

For each candidate give, in order:
  1. name
  2. problem solved
  3. outside ingredients and exact sources
  4. what borrowed unchanged
  5. what adapted for Hydra
  6. what genuinely novel synthesis
  7. why it transfers to Hydra specifically
  8. exact mathematical formulation
  9. tensor shapes and affected network interfaces
  10. exact algorithm pseudocode
  11. exact Hydra surfaces it would touch
  12. prototype path
  13. benchmark plan
  14. what success would look like
  15. what would kill idea quickly
  16. red-team failure analysis
  17. why this more likely to matter than strongest simpler mainline alternative
  18. closest known baseline and why this does not reduce to it
  19. dependency closure table: required signal / label / hook / teacher / runtime state | already exists / cheap to expose / missing | evidence or blocker
  20. minimum falsifiable prototype

Then end with:
  - single best candidate to try first
  - single best cheap benchmark to run first
  - single biggest hidden impl risk
  - 2-5 most tempting rejected directions and exact rejection reason

  ## Hard constraints

  - no generic bigger-model or bigger-search proposals
  - no AGPL-derived code advice
  - no broad architecture resets
  - no pretty but untestable ideas
  - no keeping weak options alive because they sound novel
  - no shallow acronym invention
  - no rec that cannot be written as executable objective/update rule plus pseudocode inside Hydra constraints
  - no answer that mainly summarizes literature instead of converging to prototype-worthy separator candidate
  - no candidate that survives only because prompt implicitly demanded at least one answer

Success = final answer feels like serious long-think synthesis pass and leaves at least one separator-level direction for Hydra that is mathematically explicit, hostile to self-delusion, genuinely prototypeable, and plausibly able to change Hydra's ceiling like LuckyJ's signature breakthroughs changed theirs.
]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="preserved" source_path="agent_7.md">
<![CDATA[## 1

**name**
Shift-aware support-constrained search distillation

**outside ingredients and sources**
SPIBB core idea: improve safely by bootstrapping back to baseline policy where support weak; Soft-SPIBB relaxes this into local uncertainty-aware constraints; selective calibration/abstention adds rule that rejecting uncertain targets often beats training on all targets. ([Proceedings of Machine Learning Research][1])

**why it transfers to Hydra specifically**
Hydra already wants ExIt, search-as-feature, CT-SMC belief, AFBS, and pondering; reconciliation memo says real bottleneck = loops only partly closed, not lack of advanced module names. Live encoder already has fixed search/belief/Hand-EV planes in 192×34 superset, so Hydra can amortize trusted search outputs without architecture reset. ([GitHub][2])

Hydra code unusually ready: `MjaiSample` already carries `safety_residual`, `belief_fields`, and `mixture_weights`; `mjai_loader.rs` already builds and packs those targets; `HydraTargets.policy_target` already dense; `soft_target_from_exit` already exists; `delta_q` and `safety_residual` heads already exist; loss already computes `l_delta_q`. But `sample.rs` still leaves `delta_q_target: None`, and advanced loss weights default `0.0`. Near-textbook case for “safe distillation of existing search,” not “more search everywhere.” ([GitHub][3])

**exact Hydra surfaces it would touch**
`hydra-core/src/afbs.rs` for root support/trust stats, `hydra-core/src/bridge.rs` for optional trust/debug feature export, `hydra-train/src/data/sample.rs` for filling search-derived targets, `hydra-train/src/training/losses.rs` for enabling and weighting existing losses, and `hydra-train/src/model.rs` only for consuming heads already present. [blocked]: self-play/sample-writer path that persists AFBS root stats not in supplied slice, so that shim may still need locating. ([GitHub][4])

**impl sketch or pseudocode**
Use existing hard-state gate from prior Hydra answer as first filter, then apply Soft-SPIBB-like support mixing only where search backed by visits.

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

Start with fixed calibrator, not learned one. Existing Hydra rec already gives usable thresholds (`top2_gap < 0.10`, `max risk > 0.15`, `ESS/P < 0.45`, `wall <= 12`, `root visits >= 64`, `supported-action visits >= 4`). ([GitHub][5])

**cheapest prototype path**
Do not invent new head. Reuse dense `policy_target`, populate `delta_q_target` in `sample.rs`, turn on `w_delta_q`, emit search targets only on hard-state slice above. Benchmark against 8×-budget AFBS teacher before full retrain. Pressure-tested locally with small standalone harness against real downloaded files: hook check passed, gate logic trivial to wire.

**what success would look like**
On held-out hard states, accepted subset should show materially better action-match / KL agreement to 8×-budget AFBS teacher than naive “use every searched state,” while keeping useful coverage. Then, at same runtime budget, self-play should improve in Elo / average placement without deeper online search. ([GitHub][6])

**what would kill idea quickly**
If production-budget AFBS not consistently better than base policy even on accepted subset, or accepted coverage too small to matter, stop. Deeper hidden kill = systematic teacher bias: because Hand-EV still not more realistic oracle Hydra wants and endgame still weighted PIMC rather than true exactification, method could amplify wrong search bias faster than it adds signal. ([GitHub][6])

  ## 2

**name**
Paired-scenario action racing for endgame exactification

**outside ingredients and sources**
Common random numbers reduce comparison variance when alternatives evaluated on same random scenarios; ranking-and-selection gives sequential way to allocate simulations among contenders; empirical Bernstein stopping uses variance to stop earlier than fixed-budget racing. ([arXiv][7])

**why it transfers to Hydra specifically**
Hydra architecture explicitly wants selective endgame exactification and stronger Hand-EV, but reconciliation memo says Hand-EV still heuristic and endgame still weighted PIMC rather than stronger exactification path. In code, `bridge.rs` already has `compute_ct_smc_hand_ev`, and `endgame.rs` already has `pimc_endgame_q_topk` over top-mass particles. Cheapest asymmetric win is not “more endgame search,” but “evaluate actions on same hidden scenarios, then stop when leader statistically separated.” ([GitHub][2])

**exact Hydra surfaces it would touch**
`hydra-core/src/endgame.rs` first, `hydra-core/src/ct_smc.rs` only if wanting reusable joint-scenario sampler, and later `hydra-core/src/hand_ev.rs` [blocked] once Hydra has slightly richer micro-rollout evaluator than current heuristic offensive estimate. ([GitHub][4])

**impl sketch or pseudocode**
Replace “estimate each action independently over top-mass particles” with “race top candidates on same scenario ids.”

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

This is compute-allocation change, not architecture change. In small toy simulation run here, shared-scenario pairing cut standard deviation of mean action-difference estimate by about **1.8×** and improved correct-selection rate at same sample count.

**cheapest prototype path**
Only patch `pimc_endgame_q_topk` for `wall <= 10`: same selected particle list for all actions, deterministic scenario seed from state hash, top-2 or top-3 discard candidates only, and empirical-Bernstein stop on leader vs runner-up. Do not touch AFBS yet.

**what success would look like**
At same wall-clock budget, action choice should become more stable across reruns, agreement with much higher-budget endgame evaluator should improve, and self-play uplift should concentrate in `wall <= 10` slice. ([GitHub][2])

**what would kill idea quickly**
If action-conditioned trajectories decorrelate so fast that pairing does not induce useful positive correlation, or current suffix evaluator too crude for lower-variance comparisons to matter, stop. Dies fast if variance reduction real but bias still dominant.

  ## 3

**name**
Value-of-computation ponder control

**outside ingredients and sources**
Russell–Wefald metareasoning treats computation as action whose value is expected improvement in decision quality; later MCTS metareasoning work argues computation control better viewed as ranking-and-selection problem than plain visit-allocation heuristic. ([IJCAI][8])

**why it transfers to Hydra specifically**
Hydra Final already treats opponent-turn idle time and predictive pondering as first-class edge. In code, `afbs.rs` already has `PonderResult`, `PonderCache`, predicted-child caching, and current priority rule `(0.1 - top2_gap)+risk+(1-ESS)`. Exact kind of heuristic VOC control can upgrade without increasing total search. ([GitHub][2])

**exact Hydra surfaces it would touch**
Almost entirely `hydra-core/src/afbs.rs`; optionally tiny telemetry hook in `bridge.rs` if wanting to log cache reuse and action-flip events for fitting scheduler. [blocked]: if current runtime does not already emit AFBS trace data, one light logging shim needed before fitting scheduler. ([GitHub][9])

**impl sketch or pseudocode**
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

First pass: fit `P(action_flip)` with tiny logistic model from AFBS trace data; no neural change, no extra head, no broader search.

**cheapest prototype path**
Keep existing queue and cache; only replace `compute_ponder_priority` with offline-calibrated formula using current features (`top2_gap`, `risk_score`, `particle_ess`, `visit_count`, maybe cache freshness). Then compare same total ponder budget against current heuristic. ([GitHub][9])

**what success would look like**
More cache hits that change downstream root decisions, less wasted pondering on already-settled roots, and better self-play at equal total ponder compute. Pure compute-reuse win, not bigger-search win. ([GitHub][2])

**what would kill idea quickly**
If game rarely reuses predicted-child states, or extra pondering almost never flips chosen action, VOC control has nothing to optimize and current cheap heuristic already enough.

single best candidate to try first**
Shift-aware support-constrained search distillation.

single best cheap benchmark to run first**
Take held-out hard-state set using existing Hydra triggers (`top2_gap < 0.10` or `risk > 0.15` or `ESS/P < 0.45` or `wall <= 12`), run production-budget AFBS and 8×-budget AFBS, compare naive-all-state distillation vs trust-gated distillation on accepted-state action match / KL to 8× teacher plus accepted coverage. If gated subset not clearly better at useful coverage, do not train.

single biggest hidden risk in your rec**
You may distill current AFBS bias faster than useful search signal, because Hydra’s current search teacher still sits on not-yet-realistic Hand-EV and still-simplified endgame evaluator.

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