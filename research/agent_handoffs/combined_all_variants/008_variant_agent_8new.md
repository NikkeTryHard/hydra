<combined_run_record run_id="008_invent_new_techniques" variant_id="agent_8_revised_variant_a" schema_version="1">
  <metadata>
    <notes>Substantive revised answer variant for prompt 8.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="PROMPT_8_FRESH_CONTEXT_INVENT_NEW_TECHNIQUES.md">
  <![CDATA[# Hydra fresh-context prompt — invent genuinely new techniques with math and red-team pressure

  Primary source material lives in the raw GitHub links below.

  ## Critical directive — how to read the core Hydra docs

  You must avoid a known bad behavior: fragmented keyword-peeking over large architecture docs.

  Bad behavior for this task:
  - searching for keywords first
  - reading isolated 20-100 line chunks around those keywords
  - treating the docs like logs or a grep database
  - inventing new techniques before understanding Hydra's current system holistically

  For this task, that behavior is disqualifying.

  Required reading workflow:
  1. Use your browse/fetch tool on the raw GitHub links for the core docs listed below.
  2. Read those core docs holistically and sequentially before doing narrower searching.
  3. Build a high-level model of Hydra's active path, reserve shelf, runtime structure, training surfaces, and already-partially-implemented loops.
  4. Only after that may you use narrower searching for exact details and outside inspiration.

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

  You are acting as a long-think breakthrough engineer for Hydra.

  Your job is to discover genuinely new techniques for Hydra, not merely rename known tricks or repackage existing ideas with shiny language. You may combine ingredients from different papers or systems, but only when the resulting mechanism is mathematically explicit, respects Hydra's architecture, and survives adversarial self-review.

  The target is not just novelty. The target is a separator-level breakthrough: a technique that could matter for Hydra the way LuckyJ's signature ACH/search-era breakthroughs mattered for LuckyJ. Do not imitate ACH mechanically. Instead, search for a breakthrough of comparable strategic importance for Hydra's actual architecture and bottlenecks.

  If you cannot make an idea technically crisp, kill it.

  <output_contract>
  - Return exactly the requested sections, in the requested order.
  - Be as detailed and explicit as necessary; do not optimize for brevity.
  - Return a full technical treatment, not a compressed memo.
  - Return only 1-3 serious techniques.
  - A short answer is usually a failure mode for this prompt.
  </output_contract>

  <verbosity_controls>
  - Prefer full technical exposition over compressed summary.
  - Use multi-paragraph explanations where needed.
  - Do not omit equations, derivations, tensor/interface details, pseudocode, assumptions, thresholds, edge cases, or implementation caveats when they matter.
  - When in doubt, include more mathematical detail, derivation, and mechanism detail rather than less.
  </verbosity_controls>

  <research_mode>
  - Work in 3 passes:
    1. Ingest: read the Hydra docs holistically and reconstruct the real current mainline, reserve shelf, and missing closures.
    2. Retrieve: search broadly for ingredient families, neighboring mechanisms, and counterexamples.
    3. Synthesize: keep only the techniques that are both genuinely novel for Hydra and technically viable under Hydra constraints.
  - Stop only when more searching is unlikely to change the final ranking.
  </research_mode>

  <tool_persistence_rules>
  - Prefer full-document browse/fetch for core docs over fragmented terminal-style chunk reading.
  - Search beyond the papers already surfaced when that could materially improve novelty or falsification.
  - Do not stop at the first plausible invention.
  </tool_persistence_rules>

  <calculation_validation_rules>
  - If a recommendation depends on quantitative reasoning, derive it explicitly.
  - Use executable arithmetic or small scripts when needed to sanity-check formulas, tensor shapes, threshold logic, or algorithm invariants.
  - Do not fake arithmetic that could have been checked.
  </calculation_validation_rules>

  <dependency_checks>
  - Before proposing implementation, verify Hydra already has or could cheaply expose the needed signals, labels, or runtime hooks.
  - Before proposing a new objective or target, check whether the needed trajectories, teacher outputs, or hidden-state labels actually exist or can be derived safely.
  </dependency_checks>

  <posture_reconstruction_rules>
  - Before proposing any technique, include a short "Hydra posture reconstruction" section with 5-10 bullets.
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
    - the mathematical formulation (e.g. the exact loss function or Bellman update)
    - the tensor shapes going in and out of the network
    - the exact pseudocode for the algorithm
  - If you cannot mathematically define the technique using the constraints of the Hydra architecture, you must discard the idea.
  - Do not confuse a renamed known trick with a genuinely new mechanism.
  - Be explicit about what is borrowed unchanged, what is adapted, and what is newly proposed.
  </novelty_viability_rules>

  <self_red_team_rules>
  - Before finalizing any recommendation, you must Red-Team your own ideas.
  - For every technique you propose, spend compute time actively searching the provided documents for reasons why the idea will fail.
  - Ask explicitly:
    - How does this break in a 4-player general-sum game?
    - Does this violate the partial observability constraints?
    - Does this require labels, targets, or privileged signals Hydra does not actually have?
    - Is the claimed novelty fake because the method collapses back to a known technique under Hydra's constraints?
    - Does a simpler existing Hydra path already dominate this?
  -  - Does the supposed breakthrough collapse into an incremental tuning trick once written out mathematically?
  - Only present techniques that survive this adversarial self-review.
  </self_red_team_rules>

  <anti_survey_rules>
  - Do not return a literature survey, field map, or long list of adjacent ideas without convergence.
  - Every cited outside paper, repo, or mechanism must earn its place by changing the final candidate set or the red-team analysis.
  - If a paragraph does not help define, falsify, compare, or prototype a surviving candidate, cut it.
  </anti_survey_rules>

  <novelty_honesty_rules>
  - For every surviving technique, include a "closest known baseline" subsection.
  - State the nearest known method or family, the exact overlap, and the irreducible difference.
  - If the method reduces to a known technique under realistic Hydra constraints, downgrade or reject it.
  - Label each surviving candidate as one of:
    - `A`: genuinely new mechanism
    - `B`: known mechanism with a Hydra-specific adaptation that plausibly changes capability
    - `C`: renamed or lightly modified known trick
  - Reject all `C` candidates.
  </novelty_honesty_rules>

  <minimum_falsification_rules>
  - For every surviving technique, define the minimum falsifiable prototype that tests the claimed breakthrough mechanism in isolation.
  - If the core claim cannot be tested without a large coupled rollout or major stack build-out, reject the idea as too diffuse.
  - The first benchmark should distinguish the idea from stronger tuning, more search, more data, or easier teacher signals.
  </minimum_falsification_rules>

  <completeness_contract>
  - Treat the task as incomplete until every surviving technique includes exact mechanism, mathematical formulation, tensor shapes, pseudocode, repo insertion points, cheapest prototype path, benchmark plan, and kill criteria.
  - Mark any underspecified item [blocked] rather than pretending it is ready.
  </completeness_contract>

  <verification_loop>
  - Before finalizing, verify that you actually read the core Hydra docs holistically before narrowing in.
  - Verify that each surviving technique is not just a renamed known trick.
  - Verify that each surviving technique is mathematically defined strongly enough that a coding agent could begin prototyping it.
  - Verify that the novelty claim survives your own red-team pass.
  </verification_loop>

  <dig_deeper_nudge>
  - Do not stop at the first cool invention.
  - Prefer capability-changing mechanisms over cosmetic complexity.
  - Search especially hard around belief compression, decision-focused uncertainty, teacher trust gating, opponent-mixture optimization, selective compute, multiplayer credit assignment, and privileged-to-public distillation.
  </dig_deeper_nudge>

  ## What to do

  1. Reconstruct Hydra's real current posture before inventing anything.
  2. Search for outside ingredients, neighboring ideas, and counterexamples.
  3. Invent only the smallest number of serious techniques that look genuinely new for Hydra, technically workable, and plausibly capable of producing a separator-level breakthrough rather than a minor improvement.
  4. For each candidate, write down the objective/update rule, tensor interfaces, algorithm pseudocode, repo insertion points, and cheapest falsification path.
  5. Compare each surviving candidate against the strongest simpler mainline alternative and reject it if it is mostly incremental.
  6. Try hard to kill each idea before keeping it.

  ## Deliverables

  Return only 1-3 serious techniques.

  If no candidate survives the novelty, grounding, and prototypeability filters, return `0 surviving candidates` and explain why.

  For each technique give, in order:
  1. name
  2. problem solved
  3. outside ingredients and exact sources
  4. what is borrowed unchanged
  5. what is adapted for Hydra
  6. what is genuinely novel synthesis
  7. why it fits Hydra specifically
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
  - the single best technique to try first
  - the single best cheap benchmark to run first
  - the single biggest hidden implementation risk
  - the 2-5 most tempting rejected directions and exactly why they were rejected

  ## Hard constraints

  - no generic bigger-model or bigger-search proposals
  - no AGPL-derived code advice
  - no broad architecture resets
  - no pretty but untestable inventions
  - no keeping weak options alive just because they sound novel
  - no shallow acronym invention
  - no recommendation that cannot be written as an executable objective/update rule plus pseudocode inside Hydra constraints
  - no answer that mainly summarizes literature instead of converging to a prototype-worthy separator candidate
  - no candidate that survives only because the prompt implicitly demanded at least one answer

  Success means your final answer feels like a serious long-think invention pass and leaves behind at least one genuinely new Hydra technique that is mathematically explicit, red-teamed, prototypeable, and plausibly capable of becoming a LuckyJ-caliber separator for Hydra rather than just another incremental trick.
  ]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_8new.md">
  <![CDATA[**Hydra posture reconstruction**

  * **Current mainline doctrine:** the repo’s active path is not “invent more modules,” but narrowly close the already-declared advanced-target loops. The reconciled next move is ExIt/exit, `ΔQ`, and safety-residual target generation first; belief fields, mixture weights, and opponent-hand-type stay inactive until Hydra has a credible **public-teacher** path for them. The reconciliation memo also explicitly says no new heads and no architectural expansion in that tranche. ([GitHub][1])

  * **Runtime structure is already richer than the training closure:** Hydra’s live contract is a fixed **192×34** encoder with a preserved 85-channel baseline prefix, a 46-action space, Group C dynamic search/belief features, and Group D Hand-EV features in the architectural SSOT. The project is already pointed at a two-tier 12-block actor / 24-block learner setup rather than a broad monolith. ([GitHub][2])

  * **Belief doctrine is already settled:** Hydra’s intended belief stack is unified, not duplicated—Mixture-SIB for amortized belief, CT-SMC for search-grade posterior—so proposals that create a second belief pipeline are already off-posture. ([GitHub][3])

  * **Search posture is also settled:** AFBS is supposed to be **selective / specialist / hard-state gated**, not broad default runtime search, and prior internal analysis already identified “trust-gated selective AFBS” as the surviving search-side bet. That means the next separator is more likely to come from **better teacher provenance and compute targeting** than from “deeper search everywhere.” ([GitHub][4])

  * **Hand-EV is important but not the open separator here:** Hydra’s final architecture elevates Hand-EV strongly, and prior internal analysis is explicit that Hand-EV realism comes before deeper AFBS. But that is already a known mainline closure item, not a fresh breakthrough candidate by itself. ([GitHub][5])

  * **Opponent modeling is strategically central, but heavy extensions are reserve-shelf unless promoted:** the opponent-modeling doc says Hydra’s primary differentiator is opponent modeling, while also warning that heavier extensions should be treated as reserve/future unless the reconciled doctrine promotes them. That matters because any surviving opponent-side invention has to unify with the mainline instead of re-opening speculative complexity. ([GitHub][6])

  * **The code-grounded gap is real and narrow:** in the provided raw files, `HydraOutput` already exposes `belief_fields`, `mixture_weight_logits`, `opponent_hand_type`, `delta_q`, and `safety_residual`; `HydraTargets` already has optional slots for those targets; `bridge.rs` already emits mixture weights, belief fields, entropy/ESS, and AFBS `delta_q`; and `CtSmc`, `AfbsTree`, and `robust_opponent` already contain usable runtime primitives. But the current sample/loader path still leaves `delta_q_target` and `opponent_hand_type_target` absent at collation, while the loader currently builds replay-safe safety-residual and stage-A belief targets instead. That is exactly the pattern “advanced surfaces exist, credible labels/inference loops do not.” 

  * **Current non-goals / deprioritized paths:** broad “search everywhere” AFBS, duplicated belief stacks, new heads before old heads get real labels, optimizer-theory detours, and speculative deception-style extras are all already demoted or rejected by the reconciled posture. That means weak candidates that mostly repackage those directions should be killed. ([GitHub][4])

  ---

  ## Technique 1

  **1. name**

  Posterior-consensus ExIt/ΔQ distillation with action-straddle search gating.

  **2. problem solved**

  Hydra’s next active tranche wants ExIt and `ΔQ`, but the dangerous failure mode is to distill from a **single privileged hidden world** and pretend that its action ranking is public-realizable. In Mahjong, many of the hardest states are precisely the ones where different plausible hidden worlds flip the best discard. Single-world search labels therefore leak unattainable information, while the current AFBS gate is still mostly heuristic. This technique fixes both problems at once: distill only the **posterior-stable** part of the teacher signal, and turn posterior instability into an explicit label for **where search is worth spending**. ([GitHub][1])

  **3. outside ingredients and exact sources**

  The outside ingredients that actually changed the final candidate set were: GATES, whose core idea is that privileged/self-distillation should be gated by **tutor consensus** instead of assuming privileged supervision is correct; HIB, whose core idea is that privileged-to-public distillation should reduce the oracle/student value gap; value-directed belief approximation, whose core point is that belief approximation should preserve **decision quality**, not just marginal fidelity; and calibrated value-aware model learning, which shows naive value-aware objectives can be uncalibrated and that stochastic uncertainty still matters even when deterministic values look good. Student of Games and the 2024 IJCAI look-ahead paper mattered as search-side baselines showing that guided search summaries can be distilled into a policy/value system without turning the whole stack into train-time full search. ([arXiv][7])

  **4. what is borrowed unchanged**

  Borrowed unchanged: consensus-gated distillation from GATES; privileged-to-public value-gap framing from HIB; the value-directed warning that the right compression target is **decision-relevant**, not merely distribution-close; and the general search-as-teacher pattern from Student of Games / look-ahead search distillation. ([arXiv][7])

  **5. what is adapted for Hydra**

  The adaptation is that “multiple tutor traces” become **multiple hidden-world teacher evaluations for the same public state**. The tutor is not a text model with extra context; it is a shallow Hydra teacher evaluated on (K) posterior worlds sampled from CT-SMC or, in the cheapest prototype, from the style-agnostic CT-SMC prior. The distilled target is not a final answer but a public-state action object: posterior-mean `ΔQ`, consensus-shaped exit policy, and posterior-mean safety residual. The disagreement signal is then re-used as a **search-worth** label rather than being discarded. This matches Hydra’s architecture much more closely than off-the-shelf uncertainty-weighted imitation. ([GitHub][8])

  **6. what is genuinely novel synthesis**

  The irreducible new part is the split of the teacher signal into two outputs from the **same** posterior-world ensemble:

  1. a **stable component** that Hydra should learn directly:
     [
     \bar{\Delta}(a),;\pi^*(a),;\bar r(a)
     ]

  2. an **unstable component** that Hydra should not force into the student policy, but should instead use to predict **search value-of-compute**:
     [
     c(a),;s_\epsilon,;v(a),;G(I)
     ]

  That is not just “confidence-weighted distillation.” It is a single mechanism that turns partial observability into a **teacher-abstention rule** and a **selective-compute label**.

  **7. why it fits Hydra specifically**

  Hydra already has nearly all required surfaces. The docs explicitly say belief/search targets must come from credible public teachers, not realized hidden-state labels; the active tranche already wants `exit_target`, `delta_q_target`, and safety residual; `bridge.rs` already knows how to emit mixture weights, belief fields, entropy/ESS, and AFBS `delta_q`; and the AFBS shell already has root exit policy and node-Q summaries. So this technique is not an architecture reset. It is a way to give Hydra’s already-existing advanced surfaces the **right semantics** under partial observability. ([GitHub][1])

  **8. exact mathematical formulation**

  Let (I) be a public Hydra information state, with legal mask (m \in {0,1}^{46}). Let (x_k \sim b(\cdot \mid I)), (k=1,\dots,K), be posterior worlds with normalized weights (\alpha_k), (\sum_k \alpha_k = 1).

  For each sampled world (x_k), let the teacher return:

  * (\Delta_k \in \mathbb{R}^{46}): world-conditioned action deltas, using Hydra’s existing AFBS semantics (Q(a)-Q(\text{root})) when available.
  * (\pi_k \in \Delta^{45}): teacher exit/root policy.
  * (r_k \in [0,1]^{46}): teacher safety residual or world-conditioned deal-in residual target.

  Define posterior means:
  [
  \bar{\Delta}*a = \sum*{k=1}^{K} \alpha_k \Delta_{k,a},
  \qquad
  \bar r_a = \sum_{k=1}^{K} \alpha_k r_{k,a}.
  ]

  For compatibility with the current narrow-tranche `ΔQ` scaling from prior Hydra work, define the clipped/scaled target
  [
  \tilde{\Delta}_a
  ================

  \mathrm{clip}!\left(\frac{\bar{\Delta}*a}{s*\Delta}, -1, 1\right),
  \qquad s_\Delta = 0.15.
  ]
  I kept (s_\Delta=0.15) because it matches the current narrow-tranche `ΔQ` normalization proposal rather than inventing a new scale. ([GitHub][1])

  Define posterior top-action mass:
  [
  c_a
  ===

  \sum_{k=1}^{K}
  \alpha_k,
  \mathbf 1!\left[
  a = \arg\max_{b:m_b=1}\Delta_{k,b}
  \right].
  ]

  Define posterior variance:
  [
  v_a
  ===

  \sum_{k=1}^{K}
  \alpha_k,
  (\Delta_{k,a} - \bar{\Delta}_a)^2.
  ]

  Let
  [
  a_1 = \arg\max_{a:m_a=1}\bar{\Delta}*a,
  \qquad
  a_2 = \arg\max*{a\neq a_1,;m_a=1}\bar{\Delta}*a.
  ]
  Define action-straddle mass:
  [
  s*\epsilon
  ==========

  \sum_{k=1}^{K}
  \alpha_k,
  \mathbf 1!\left[
  \Delta_{k,a_2} \ge \Delta_{k,a_1} - \epsilon_{\mathrm{flip}}
  \right].
  ]

  For the prototype, I sanity-checked (\epsilon_{\mathrm{flip}}=0.05), (\tau_c=0.70), (\tau_s=0.35), and (\tau_v=0.05) on synthetic stable / top-two-flip / high-variance cases; they cleanly separated those toy cases, but they are only seed thresholds, not validated production constants.

  Define teacher reliability for **distillation**:
  [
  \lambda_{\mathrm{teach}}
  ========================

  \mathrm{clip}!\left(
  \frac{\max_a c_a - \tau_c}{1-\tau_c},,0,,1
  \right)
  \cdot
  \exp!\left(
  -\frac{\max_a v_a}{\tau_v}
  \right)
  \cdot
  \exp!\left(
  -\frac{s_\epsilon}{\tau_s}
  \right)
  \cdot
  \chi_{\mathrm{ESS}},
  ]
  where
  [
  \chi_{\mathrm{ESS}}
  ===================

  \mathrm{clip}!\left(
  \frac{\mathrm{ESS}/P - \tau_{\mathrm{ess}}}{1-\tau_{\mathrm{ess}}},
  0,1
  \right)
  ]
  if CT-SMC ESS is available, and (\chi_{\mathrm{ESS}}=1) in the cheapest prior-only prototype.

  Define the consensus-shaped exit target:
  [
  \pi^*_a
  =======

  \frac{
  m_a,
  \exp(\bar{\Delta}*a/\tau*\pi),
  (c_a+\epsilon_c)^\gamma
  }{
  \sum_b
  m_b,
  \exp(\bar{\Delta}*b/\tau*\pi),
  (c_b+\epsilon_c)^\gamma
  }.
  ]

  For **search-worth**, do not threshold instability directly. Instead compute the posterior expected regret reduction from switching the current base action to the ensemble-best action:
  [
  a_{\mathrm{base}}
  =================

  \arg\max_{a:m_a=1}
  \Big(
  \ell_\theta(a\mid I)
  +
  \beta_\Delta \hat{\Delta}_\theta(a\mid I)
  \Big),
  ]
  [
  a^* = a_1,
  ]
  [
  G(I)
  ====

  \sum_{k=1}^{K}
  \alpha_k
  \left[
  \Delta_{k,a^*}
  --------------

  \Delta_{k,a_{\mathrm{base}}}
  \right].
  ]

  Then define the search-gate label as
  [
  y_{\mathrm{search}}
  ===================

  \mathbf 1!\left[
  G(I) > \delta_g
  \right].
  ]
  For the prototype, (\delta_g) should be chosen as a held-out quantile of the empirical (G(I)) distribution—e.g. top quartile—rather than an unvalidated absolute constant.

  Losses:

  Policy / ExIt:
  [
  L_{\pi}
  =======

  \lambda_{\mathrm{teach}}
  \cdot
  \mathrm{KL}\big(\pi^* ,|, \hat{\pi}_\theta\big).
  ]

  Delta-Q:
  [
  L_{\Delta}
  ==========

  \lambda_{\mathrm{teach}}
  \cdot
  \frac{
  \sum_a m_a,\mathrm{Huber}!\left(
  \hat{\Delta}_\theta(a)-\tilde{\Delta}_a
  \right)
  }{
  \max(1,\sum_a m_a)
  }.
  ]

  Safety residual:
  [
  L_{\mathrm{safety}}
  ===================

  \lambda_{\mathrm{teach}}
  \cdot
  \frac{
  \sum_a m_a,\mathrm{Huber}!\left(
  \hat r_\theta(a)-\bar r_a
  \right)
  }{
  \max(1,\sum_a m_a)
  }.
  ]

  Gate:
  [
  L_{\mathrm{gate}}
  =================

  \mathrm{BCE}!\left(g_\psi(z(I)),, y_{\mathrm{search}}\right),
  ]
  with (z(I)) the public/search feature vector defined below.

  **9. tensor shapes and affected network interfaces**

  Main model outputs are unchanged in the cheapest prototype:

  * `policy_logits`: ([B,46])
  * `delta_q`: ([B,46])
  * `safety_residual`: ([B,46])

  These shapes already exist in Hydra’s provided model and loss surfaces.

  Teacher ensemble tensors per sample:

  * world deltas: ([K,46])
  * world safety residuals: ([K,46])
  * world weights: ([K])
  * posterior-optimal-action masses (c): ([46])
  * posterior variances (v): ([46])
  * search-worth scalar (G(I)): ([1])

  New dataset-side tensors:

  * `delta_q_target`: ([B,46]) — already supported train-side, currently unfilled in the provided sample path.
  * `exit_target`: ([B,46]) — upstream production is explicitly called out as missing-but-desired in the reconciliation path. ([GitHub][8])
  * `safety_residual_target`: ([B,46]) — already live replay-side in the current loader.
  * `search_need_label`: ([B,1]) — new auxiliary dataset field for the gate, not a new Hydra head.

  Gate feature vector:
  [
  z(I)\in\mathbb{R}^9
  ]
  with prototype features:

  1. base-policy top-2 gap,
  2. predicted `delta_q` top-2 gap,
  3. mixture entropy,
  4. mixture ESS,
  5. max opponent tenpai probability,
  6. danger of the current top action,
  7. wall fraction remaining,
  8. orasu flag,
  9. score-gap-to-next-rank.

  Everything in that vector is already present or cheap to expose from current bridge/model/runtime state. ([GitHub][5])

  **10. exact algorithm pseudocode**

  ```text
  OFFLINE_LABEL_BUILD(I, model_snapshot):
      worlds = SAMPLE_POSTERIOR_WORLDS(I, K)     # CT-SMC if available; prior-only CT-SMC for cheapest proto
      for each (x_k, alpha_k) in worlds:
          delta_k, exit_k, safety_k = WORLD_TEACHER(I, x_k)
          store delta_k, exit_k, safety_k

      bar_delta[a] = sum_k alpha_k * delta_k[a]
      bar_safety[a] = sum_k alpha_k * safety_k[a]

      c[a] = sum_k alpha_k * 1[a == argmax_legal(delta_k)]
      v[a] = sum_k alpha_k * (delta_k[a] - bar_delta[a])^2

      a1 = argmax_legal(bar_delta)
      a2 = second_best_legal(bar_delta)
      s_eps = sum_k alpha_k * 1[delta_k[a2] >= delta_k[a1] - eps_flip]

      lambda_teach = distill_reliability(c, v, s_eps, ess_ratio(worlds))
      pi_star = consensus_shaped_exit(bar_delta, c, legal_mask(I))

      a_base = argmax_legal(policy_logits(model_snapshot, I) + beta_delta * pred_delta(model_snapshot, I))
      G = sum_k alpha_k * (delta_k[a1] - delta_k[a_base])
      y_search = 1[G > delta_g_quantile]

      emit:
          delta_q_target = clip(bar_delta / s_delta, -1, 1)
          exit_target = pi_star
          safety_residual_target = bar_safety
          target_weight = lambda_teach
          search_need_label = y_search
          gate_features = z(I)
  ```

  ```text
  TRAIN_STEP(batch):
      out = hydra_model.forward(batch.obs)

      L = core_policy_value_losses(out, batch)

      if batch.delta_q_target present:
          L += w_delta_q * weighted_masked_huber(out.delta_q, batch.delta_q_target, batch.legal_mask, batch.target_weight)

      if batch.exit_target present:
          L += w_exit * weighted_kl(log_softmax(out.policy_logits), batch.exit_target, batch.target_weight)

      if batch.safety_residual_target present:
          L += w_safety * weighted_masked_huber(out.safety_residual, batch.safety_residual_target, batch.legal_mask, batch.target_weight)

      gate_logit = gate_mlp(batch.gate_features)
      L += w_gate * BCEWithLogits(gate_logit, batch.search_need_label)

      update(L)
  ```

  ```text
  RUNTIME_DECISION(I):
      out = hydra_model.forward(I.obs)
      z = build_gate_features(I, out, mixture, ct_smc, score_ctx)

      if gate_mlp(z) > tau_gate:
          run_selective_afbs(I)
      else:
          skip_search()
  ```

  **11. exact Hydra surfaces it would touch**

  * `hydra-core/src/ct_smc.rs`: use existing world sampler / ESS outputs for posterior worlds.
  * `hydra-core/src/afbs.rs`: expose the shallow world teacher object used in offline label generation; optionally replace or augment `compute_ponder_priority` with learned gate output. 
  * `hydra-core/src/bridge.rs`: export gate features (z), since it already aggregates mixture weights, belief fields, entropy/ESS, and AFBS `delta_q`.
  * `hydra-train/src/data/sample.rs`: add `delta_q_target`, `exit_target`, `search_need_label`, `gate_features`, and per-sample `target_weight`. `delta_q_target` is already supported train-side but currently not collated in the provided sample path.
  * `hydra-train/src/data/mjai_loader.rs`: add the offline hard-state replay walker that reconstructs public states and builds posterior-world teacher labels. ([GitHub][8])
  * `hydra-train/src/training/losses.rs`: reuse existing masked action losses and optional target slots; add weighted exit KL if needed.
  * `hydra-train/src/training/bc.rs` / `rl.rs`: wire upstream `exit_target` production, which the reconciliation memo explicitly says is missing and belongs in the tranche. ([GitHub][8])

  **12. prototype path**

  Use the cheapest path that still tests the mechanism itself:

  1. restrict to **discard decisions on hard-state slices**,
  2. use (K=8) posterior worlds sampled from the existing CT-SMC prior / public-count-consistent sampler,
  3. use a **shallow world teacher** instead of full AFBS—e.g. current Hand-EV + exact deal-in risk + score-context scalarization,
  4. train only:

     * `delta_q`,
     * policy KL to `exit_target`,
     * gate MLP.

  This keeps the first test independent of a fully mature AFBS teacher and independent of any new opponent-style model. The mechanism is falsified if it fails even with that cheap setup.

  I also checked the cost arithmetic. If the current searched hard state budget is roughly “1 world × 128 visit-equivalents,” then “8 worlds × 16 visit-equivalents” costs about **1.21× to 1.78× per searched hard state** when world-init overhead is 4–16 visit-equivalents. If only 5% of states are searched, that is only about **1.01× to 1.04× overall**. So the ensemble teacher is not automatically too expensive if it redistributes existing hard-state compute instead of blindly multiplying it.

  **13. benchmark plan**

  Offline first.

  Primary metric:
  [
  \mathrm{PER}(\hat a)
  ====================

  \sum_k \alpha_k
  \left[
  \max_a \Delta_k(a) - \Delta_k(\hat a)
  \right],
  ]
  posterior expected regret of the chosen action.

  Compare three labelers at fixed teacher compute:

  * **Baseline A:** single exact hidden-world teacher,
  * **Baseline B:** posterior-mean teacher without consensus shaping,
  * **Candidate:** posterior-consensus teacher + learned search gate.

  Report:

  * posterior expected regret,
  * policy agreement with posterior-mean best action,
  * gate AUROC / PR-AUC for (y_{\mathrm{search}}),
  * fraction of AFBS calls spent on genuinely high-(G(I)) states,
  * duplicate online delta at fixed search budget.

  The key control is fixed total teacher compute. If candidate wins only because it uses more search, it fails the separator claim.

  **14. what success would look like**

  Success is:

  * lower posterior expected regret than both baselines at the same teacher budget,
  * a gate that clearly beats the current heuristic gate on predicting actual search gain,
  * online improvement at the same AFBS budget because search is concentrated on the states where hidden-world disagreement actually matters,
  * and, critically, better stability of `ΔQ` / ExIt supervision rather than just more labels.

  **15. what would kill the idea quickly**

  Kill it if any of these happen:

  * posterior-consensus targets collapse to almost the same thing as ordinary posterior-mean targets,
  * the learned gate adds no predictive lift over current heuristic AFBS gating,
  * gains disappear when teacher compute is matched,
  * the cheap posterior-world sampler is too noisy and makes consensus weights meaningless,
  * or the online win comes only from the gate spending more search, not from better label semantics.

  **16. red-team failure analysis**

  How it breaks in a 4-player general-sum game: posterior expectation of one-step action deltas is not a multiplayer equilibrium object. That is true. The response is to measure success by **posterior expected regret** and duplicate match results, not by pretending the target is a solved game-theoretic value.

  How it could violate partial observability: if you distill single hidden worlds, you absolutely leak privileged information. This technique only survives because it distills the **posterior-stable** component and turns the unstable component into abstention / search labels. If someone simplifies it back to “average the worlds and train harder,” the novelty mostly disappears.

  How it could depend on unavailable labels: the full version needs an offline posterior-world label builder. That builder is not fully wired today. But the provided code already has CT-SMC, AFBS root summaries, and replay-safe loader machinery, so the missing part is integration, not an impossible dependency. 

  How the novelty could be fake: if the gate is just a logistic on existing heuristics and the world-ensemble distillation is dropped, it reduces to “better AFBS gating,” which is incremental. If the gate is kept but the target remains single-world privileged, it reduces to ordinary trust-gated search supervision. The candidate only survives as the **joint** mechanism.

  How a simpler Hydra path might dominate it: the strongest simpler path is the existing narrow-tranche `ExIt + ΔQ + safety_residual` closure with heuristic hard-state gating. If that already captures almost all recoverable search gain, this candidate is not separator-level and should be rejected. ([GitHub][1])

  **17. why this is more likely to matter than the strongest simpler mainline alternative**

  The strongest simpler alternative is exactly the current Hydra mainline: wire `exit_target`, `delta_q_target`, and safety residual, then use a heuristic hard-state gate. That closes loops, but it still trains on labels that can be **wrong for the public state** when hidden-world action rankings flip. This candidate attacks the specific source of that error. It is more likely to matter if Hydra’s real ceiling blocker is “search labels are semantically misaligned under partial observability,” not “we just need more labels.” ([GitHub][1])

  **18. closest known baseline and why this does not reduce to it**

  Closest known baseline: consensus-gated privileged distillation / uncertainty-weighted distillation.

  Exact overlap: use disagreement to reduce trust in privileged supervision. ([arXiv][7])

  Irreducible difference: here the disagreement is **posterior hidden-world action disagreement for one public state**, not model disagreement or trace disagreement, and the same posterior-world ensemble is used both to shape the action target and to create a **search-gain** label. That extra coupling is why I classify it as **B**: a known mechanism family with a Hydra-specific adaptation that plausibly changes capability.

  **19. dependency closure table**

  | required signal / label / hook / teacher / runtime state | already exists / cheap to expose / missing | evidence or blocker                                                                                                                                                              |
  | -------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | 192×34 public observation tensor                         | already exists                             | `TESTING.md` status note says live contract is 192×34. ([GitHub][2])                                                                                                             |
  | 46-action legal mask                                     | already exists                             | Game-engine action contract is fixed at 46 actions. ([GitHub][9])                                                                                                                |
  | CT-SMC particle sampler + ESS                            | already exists                             | `CtSmcConfig`, ESS thresholding, weighted tile counts already exist.                                                                                                             |
  | AFBS root policy / node-Q summaries                      | already exists                             | `root_exit_policy`, `node_q_value`, ponder shell already exist.                                                                                                                  |
  | `delta_q_target` train-side slot                         | already exists                             | `HydraTargets` already includes it.                                                                                                                                              |
  | `safety_residual_target` train-side slot                 | already exists                             | `HydraTargets` already includes it; loader already builds replay-side safety residual.                                                                                           |
  | upstream `exit_target` production                        | cheap to expose / partially missing        | reconciliation says upstream production belongs in the tranche; current gap is integration. ([GitHub][8])                                                                        |
  | `delta_q_target` collation in `MjaiBatch`                | missing currently                          | provided sample path leaves `delta_q_target` absent.                                                                                                                             |
  | gate feature vector (z)                                  | cheap to expose                            | bridge already emits mixture weights, belief fields, entropy/ESS, AFBS `delta_q`, and risk/stress hooks.                                                                         |
  | posterior-world offline label builder                    | missing                                    | no explicit offline generation path is wired today; reconciliation warns not to silently couple replay loading to runtime-only search context without such a path. ([GitHub][8]) |

  **20. minimum falsifiable prototype**

  Take a hard-state replay slice of discard decisions. For each public state:

  * sample (K=8) CT-SMC-consistent worlds,
  * compute a shallow world-conditioned discard score (Hand-EV + exact deal-in risk + score-context term),
  * build posterior-consensus `ΔQ` / exit targets,
  * label search-worth from posterior expected regret reduction (G(I)),
  * train `delta_q` and a tiny gate MLP only.

  Benchmark against:

  * single-world teacher,
  * posterior-mean teacher without consensus shaping,
  * current heuristic gate.

  If posterior expected regret and gate AUROC do not improve at fixed teacher compute, kill the idea immediately.

  ---

  ## Technique 2

  **1. name**

  Archetype-coupled CT-SMC via Rao-Blackwellized opponent regime posteriors.

  **2. problem solved**

  Hydra currently has three partially separate objects:

  * hidden-tile posterior (Mixture-SIB / CT-SMC),
  * opponent-hand-type / archetype surfaces,
  * robust-opponent backup math.

  The open problem is that public opponent actions should update **all three together**. A discard or call is evidence about concealed tiles, but the meaning of that evidence depends on opponent style / plan. Keeping style and hidden tiles separate throws away exactly the signal Hydra claims as its primary differentiator. This technique makes opponent regime posterior a first-class part of the CT-SMC state so that style evidence sharpens tile beliefs and tile beliefs sharpen style evidence. ([GitHub][6])

  **3. outside ingredients and exact sources**

  Two outside sources materially mattered here. Differentiable Interacting Multiple Model Particle Filtering contributed the concrete idea of maintaining regime probabilities alongside particles and letting regime probability guide computation. The multiplayer opponent-modeling paper mattered because it is a direct counterexample to the lazy equilibrium-only posture: in multiplayer imperfect-information games, using observations of opponents can outperform equilibrium-style play. ([arXiv][10])

  **4. what is borrowed unchanged**

  Borrowed unchanged: the IMM idea that sequential inference can maintain a small regime posterior jointly with particle state, and the generic idea that regime probability can be updated by action likelihoods and used to allocate downstream compute. ([arXiv][10])

  **5. what is adapted for Hydra**

  The adaptation is that the particle state is not a dynamical-system latent but a Mahjong hidden-tile contingency allocation. The regimes are not motion models but **opponent strategic archetypes / public-teacher hand-type modes**. The emission model is not a sensor likelihood but a likelihood of an opponent’s **public Mahjong action** given a particle’s concealed hand allocation and a regime. The resulting posterior is then fed directly into Hydra’s existing robust-opponent code and into the dormant `opponent_hand_type` supervision surface. ([GitHub][6])

  **6. what is genuinely novel synthesis**

  The novel synthesis is not just “add opponent clusters.” It is the closed Bayesian loop:

  1. the public prior over regimes comes from Hydra’s public encoder,
  2. each CT-SMC particle carries a per-opponent regime posterior,
  3. public actions update both particle weight and regime posterior,
  4. the aggregated regime posterior becomes:

     * a soft teacher for `opponent_hand_type`,
     * a conditioning object for robust search backup,
     * and a better hidden-world posterior for any belief/search target downstream.

  That collapses three disconnected Hydra shelves into one object.

  **7. why it fits Hydra specifically**

  Hydra is unusually ready for this:

  * CT-SMC is already present,
  * `SearchContext` already has optional `CtSmc`, `MixtureSib`, and opponent-risk/stress hooks,
  * `robust_opponent.rs` already contains `robust_backup` and `archetype_softmin`,
  * and the model/loss surfaces already include `opponent_hand_type`, even though the current code path does not yet produce a credible target for it. ([GitHub][1])

  That is why this is worth keeping despite being more ambitious than Technique 1: it aligns with Hydra’s intended final identity rather than inventing a side project.

  **8. exact mathematical formulation**

  Let (x) denote the hidden tile allocation and (r_j \in {1,\dots,R}) the latent regime for opponent (j \in {1,2,3}).

  Maintain (P) particles:
  [
  {x^{(m)}, w^{(m)}, \rho^{(m)}*{1}, \rho^{(m)}*{2}, \rho^{(m)}*{3}}*{m=1}^{P},
  ]
  where each
  [
  \rho^{(m)}_{j} \in \Delta^{R-1}
  ]
  is a categorical posterior over regimes for opponent (j) inside particle (m).

  Public prior from Hydra’s public model or a stand-alone prior net:
  [
  q_\eta(r_j \mid H^{pub}*{j,t}).
  ]
  Initialize:
  [
  \rho^{(m)}*{j,0} = q_\eta(\cdot \mid H^{pub}_{j,t})
  \quad \forall m.
  ]

  Let (o_t^j) be the observed public action from opponent (j) at time (t). Let the emission model be
  [
  \ell_\phi(o_t^j \mid I_t, x^{(m)}, r)
  =====================================

  p_\phi(o_t^j \mid \phi(I_t, x^{(m)}_j), r),
  ]
  where (\phi(I_t, x^{(m)}_j)) is a feature extractor over public state plus the concealed-hand summary for opponent (j) implied by particle (m).

  Rao-Blackwellized update for one acting opponent (j):

  [
  L_j^{(m)}
  =========

  \sum_{r=1}^{R}
  \rho_{j,r}^{(m)}
  \ell_\phi(o_t^j \mid I_t, x^{(m)}, r),
  ]

  [
  w^{(m)}
  \leftarrow
  \frac{
  w^{(m)} L_j^{(m)}
  }{
  \sum_{m'}
  w^{(m')} L_j^{(m')}
  },
  ]

  [
  \rho_{j,r}^{(m)}
  \leftarrow
  \frac{
  \rho_{j,r}^{(m)}
  \ell_\phi(o_t^j \mid I_t, x^{(m)}, r)
  }{
  L_j^{(m)}
  }.
  ]

  If multiple opponents act before the next filter step, multiply the corresponding (L_j^{(m)}) factors.

  Optional regime drift:
  [
  \rho_j^{(m)} \leftarrow T^\top \rho_j^{(m)}
  ]
  before the update, with (T=I) in the first prototype.

  Aggregate regime posterior:
  [
  \bar{\rho}_{j,r}
  ================

  \sum_{m=1}^{P}
  w^{(m)} \rho_{j,r}^{(m)}.
  ]

  This becomes the soft teacher target:
  [
  y^{type}*{j,r} = \bar{\rho}*{j,r}.
  ]

  Training objective for the public prior and emission model on replay events with exact hidden states:
  [
  L_{\mathrm{emit}}
  =================

  -\sum_{e}
  \log
  \sum_{r=1}^{R}
  q_\eta(r \mid H^{pub}*e),
  \ell*\phi(o_e \mid I_e, x_e, r).
  ]

  To avoid trivial regime collapse, add a mild balance regularizer:
  [
  L_{\mathrm{bal}}
  ================

  \lambda_{\mathrm{bal}}
  D_{KL}
  \left(
  \frac{1}{N}\sum_e q_\eta(\cdot \mid H^{pub}_e)
  ;\middle|;
  \mathrm{Unif}(R)
  \right).
  ]

  Total training loss:
  [
  L = L_{\mathrm{emit}} + L_{\mathrm{bal}}.
  ]

  Search integration: if (Q_r(a)) is the action value under regime (r), use Hydra’s existing robust-opponent machinery:
  [
  \tilde{Q}(a)
  ============

  \min_{q \in \Delta^R,; D_{KL}(q | \bar{\rho}*j)\le \epsilon_j}
  \sum*{r=1}^{R} q_r Q_r(a).
  ]
  That is exactly the KL-ball robust backup Hydra already has math for.

  Optional compute allocation per regime, adapted from IMM:
  [
  N_r
  ===

  \max(N_{\min},
  \left\lfloor
  P \cdot
  \frac{\bar{\rho}*r^\alpha (U_r+\varepsilon)^\beta}
  {\sum*{r'} \bar{\rho}*{r'}^\alpha (U*{r'}+\varepsilon)^\beta}
  \right\rfloor),
  ]
  where (U_r) can be within-regime entropy or ESS deficit. I would **not** put this in the minimum prototype.

  **9. tensor shapes and affected network interfaces**

  Runtime sidecar state:

  * existing particle allocation: ([P,34,4])
  * new regime sidecar: ([P,3,R])

  At (P=128):

  * (R=4) adds (128\times3\times4 = 1536) float32s (\approx 6) KiB,
  * (R=8) adds (3072) float32s (\approx 12) KiB.

  So memory overhead is negligible.

  Network side:

  * reuse `opponent_hand_type` as the public prior head:
    [
    [B, 3R].
    ]
    The current code already makes the class count configurable via `opponent_hand_type_classes`. If checkpoint compatibility matters, keep (R=8); if prototype simplicity matters, set (R=4). The parameter delta from (R=8\to4) at hidden size 256 is only **3084 parameters**, so this is not an architecture crisis.

  Train-side targets:

  * `opponent_hand_type_target`: ([B,3R])
  * optional `opponent_hand_type_mask`: ([B,3R])

  Emission model:

  * discard-only prototype input:
    [
    \phi \in \mathbb{R}^{F_{\mathrm{opp}}},
    \quad F_{\mathrm{opp}} \approx 64
    ]
    [inference: concealed hand histogram + shanten / wait / yaku flags + public score/wall context],
  * output logits:
    [
    [B, R, 34]
    ]
    for discard-tile likelihoods in the discard-only prototype.

  Per observed opponent action, the number of regime-likelihood evaluations is only (P \times R) for one acting opponent: 512 at (P=128,R=4), or 1024 at (P=128,R=8). Even three simultaneous acting-opponent updates remain small.

  **10. exact algorithm pseudocode**

  ```text
  # Phase A: fit public prior + regime-conditioned emission model on replay
  for event e in replay_opponent_events:
      q = prior_model(pub_history_e)                       # [R]
      for r in 1..R:
          loglik[r] = emission_model(pub_ctx_e, exact_hidden_hand_e, r, observed_action_e)

      loss = -logsumexp(log(q) + loglik) + lambda_bal * KL(mean_batch(q), uniform_R)
      update(prior_model, emission_model)
  ```

  ```text
  # Phase B: runtime / offline coupled filter update
  for particle m in 1..P:
      for acting opponent j:
          for r in 1..R:
              l[r] = emission_model(pub_ctx_t, hidden_hand_from_particle(m, j), r, observed_action_t_j)

          L = dot(rho[m][j], l)
          rho[m][j] = rho[m][j] * l / L
          w[m] = w[m] * L

  normalize(w)

  if ESS(w) < ess_threshold:
      resample particles
      copy rho sidecars along ancestry

  for opponent j:
      bar_rho[j] = sum_m w[m] * rho[m][j]
  ```

  ```text
  # Phase C: train soft opponent-hand-type target
  for public state I in training batch:
      out = hydra_model.forward(I.obs)
      target = concat_j(bar_rho[j])          # [3R]
      loss += w_type * KL(target || softmax(out.opponent_hand_type))
  ```

  ```text
  # Phase D: search-side robust backup
  for opponent node j:
      q_per_arch = archetype_conditioned_action_values(node, j)   # [R][A]
      q_robust = archetype_softmin(q_per_arch, bar_rho[j], tau_arch or KL-ball epsilon)
      use q_robust in backup
  ```

  **11. exact Hydra surfaces it would touch**

  * `hydra-core/src/ct_smc.rs`: extend particle state or add a parallel sidecar for (\rho[P,3,R]); reuse the existing update / ESS / resampling path.
  * `hydra-core/src/bridge.rs`: export (\bar\rho), regime entropy, and optionally regime-conditioned stress/risk summaries. `SearchContext` already has the right neighborhood for this.
  * `hydra-core/src/robust_opponent.rs`: consume (\bar\rho) directly via `robust_backup` / `archetype_softmin`.
  * `hydra-core/src/afbs.rs`: optional root-only robust opponent backup first; full opponent-node semantics later. 
  * `hydra-train/src/model.rs`: reuse the existing configurable `opponent_hand_type` head as public regime prior.
  * `hydra-train/src/training/losses.rs`: activate `opponent_hand_type_target` loss once credible labels exist.
  * `hydra-train/src/data/sample.rs`: collate `opponent_hand_type_target`, which is currently absent in the provided sample path.
  * new module `[cheap new file]` such as `hydra-core/src/opponent_emission.rs`: the only genuinely missing runtime surface.

  **12. prototype path**

  Do not start with the full joint monster. Start with a discard-only falsifier:

  1. choose (R=4),
  2. train a public prior + regime-conditioned **discard** likelihood model on replay events with exact hidden hands,
  3. add (\rho[P,3,R]) sidecars to offline CT-SMC only,
  4. update particle weights from observed discards only,
  5. evaluate whether hidden-world posterior calibration improves,
  6. only then turn (\bar\rho) into `opponent_hand_type_target`,
  7. only after that test root-only robust backup.

  This keeps the first test focused on the actual claim: style-coupled filtering sharpens posterior belief.

  **13. benchmark plan**

  First benchmark: filtering quality, not Elo.

  Compare:

  * style-agnostic CT-SMC,
  * CT-SMC + public prior only,
  * candidate coupled filter.

  Hold particles fixed.

  Report:

  * held-out opponent discard NLL,
  * wait-set calibration / exact-wait recall,
  * posterior log-likelihood of true concealed tile membership,
  * `opponent_hand_type` ECE / KL once that head is trained.

  Second benchmark: fixed-state search slices.

  * Use the same hard states with the same particle count and AFBS budget.
  * Compare action quality / posterior expected regret with and without the coupled filter.

  Third benchmark: duplicate online play.

  * Same AFBS visits, same particle count, same evaluation seed bank.

  **14. what success would look like**

  Success is a clean offline win before any large online claim:

  * better held-out discard likelihood,
  * better wait / concealed-tile posterior calibration,
  * a nontrivial, well-calibrated (\bar\rho) rather than near-uniform mush,
  * and then a root-level action improvement at fixed compute when robust backup is allowed to use (\bar\rho).

  If those do not happen, there is no reason to believe the full search-side version will matter.

  **15. what would kill the idea quickly**

  Kill it if:

  * the regime model collapses to one mode,
  * the coupled filter does not improve posterior calibration over style-agnostic CT-SMC,
  * (\bar\rho) remains too diffuse to change any robust backup,
  * or gains disappear once particle count and AFBS visits are matched.

  **16. red-team failure analysis**

  How it breaks in a 4-player general-sum game: exploitive style inference can backfire badly if the regime posterior is overconfident or if opponents are nonstationary. The right antidote is exactly Hydra’s own robust-opponent math: treat (\bar\rho) as the center of a KL ball, not a perfect point estimate. ([arXiv][11])

  How it could violate partial observability: using particle hidden hands in the emission likelihood is fine; that is ordinary Bayesian filtering. The failure mode is not “illegitimate hidden info,” it is **overconfident likelihoods** that collapse particle weights. That is why the emission model needs calibration and why the first prototype should be offline only.

  How it depends on missing surfaces: the major missing piece is the opponent-action emission model. That is real. Everything else is either already in the provided code or one level of integration away.

  How the novelty could be fake: if (\rho) depends only on public history and does **not** feed back into particle weights, this collapses to a plain public-only opponent classifier. If (\rho) updates particle weights but is never used in robust search or supervision, it becomes just a fancy filter variant. The full claim only survives when the same posterior drives **belief**, **training target**, and **search backup**.

  How a simpler Hydra path might dominate it: the simpler path is “train `opponent_hand_type` from a public-only classifier and use fixed archetype weights in robust backup.” If that works just as well, this candidate is too fancy. That is why the offline posterior-calibration benchmark is the kill switch.

  **17. why this is more likely to matter than the strongest simpler mainline alternative**

  The strongest simpler alternative is a public-only opponent-hand-type head plus a later robust residual. That leaves a major source of Bayesian signal unused: an action’s meaning depends on the concealed hand, and the concealed-hand posterior should change when a style hypothesis changes. The candidate is more likely to matter if Hydra’s edge really is “read opponents better than everyone else,” because it is the first mechanism here that makes opponent reading update the hidden-world posterior directly rather than living as a separate side prediction. ([GitHub][6])

  **18. closest known baseline and why this does not reduce to it**

  Closest known baseline: interacting-multiple-model particle filtering / mixture-of-experts sequential inference.

  Exact overlap: multiple regimes, sequential regime posterior update, regime-guided compute. ([arXiv][10])

  Irreducible difference: the regime latent is attached to a Mahjong hidden-tile allocation particle filter, and the resulting posterior is reused for **public-teacher soft labels** and **robust search backup**. That makes it a **B** candidate, not an **A**: the core family is known, but the Hydra-specific coupling is capability-relevant and does not collapse into an ordinary opponent classifier.

  **19. dependency closure table**

  | required signal / label / hook / teacher / runtime state | already exists / cheap to expose / missing | evidence or blocker                                                                                                                               |
  | -------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
  | CT-SMC particle state + ESS / resampling                 | already exists                             | `CtSmc` already has particles, update, ESS, weighted counts.                                                                                      |
  | public search context hooks for risk / stress            | already exists                             | `SearchContext` already includes opponent-risk / stress hooks.                                                                                    |
  | robust KL-ball / archetype softmin math                  | already exists                             | `robust_backup`, `archetype_softmin`, calibration helpers already exist.                                                                          |
  | public prior head over opponent modes                    | already exists / configurable              | `opponent_hand_type` head and configurable class count already exist.                                                                             |
  | loss slot for opponent-hand-type soft target             | already exists                             | `HydraTargets` and `HydraLossConfig` already include it.                                                                                          |
  | sample / batch collation for `opponent_hand_type_target` | missing currently                          | provided sample path leaves it as `None`.                                                                                                         |
  | replay-hidden opponent hand reconstruction               | cheap to expose / mostly already there     | loader already reconstructs exact waits / hidden-state-derived danger signals; extending that to opponent-action events is plausible [inference]. |
  | opponent-action emission model (p(o\mid I,x,r))          | missing                                    | this is the main new module; no direct implementation surface exists yet                                                                          |
  | search-side consumption of (\bar\rho)                    | cheap to expose later                      | robust-opponent utilities and search context hooks already exist.                                                                                 |

  **20. minimum falsifiable prototype**

  Train a discard-only regime-conditioned emission model on replay events with exact hidden hands. Then run offline CT-SMC on held-out states with and without the (\rho[P,3,R]) sidecar and compare posterior calibration at the same particle count.

  If discard NLL, wait recall, and concealed-tile posterior quality do not improve, kill the idea before touching AFBS or the main Hydra model.

  ---

  * **the single best technique to try first:** Technique 1 — posterior-consensus ExIt/ΔQ distillation with action-straddle search gating.

  * **the single best cheap benchmark to run first:** an offline hard-state discard suite where the score is **posterior expected regret** at fixed teacher compute, comparing single-world labels, posterior-mean labels, and posterior-consensus labels, plus gate AUROC on search-worth (G(I)).

  * **the single biggest hidden implementation risk:** posterior quality itself. Both surviving candidates depend on the posterior representing the **right** public uncertainty object; if Hydra’s current style-agnostic posterior is too blurry or miscalibrated, Technique 1 can abstain on the wrong states and Technique 2 can confidently sharpen the wrong worlds.

  * **the 2-5 most tempting rejected directions and exactly why they were rejected:**

    * **Gauge-fixed / projected belief supervision alone:** important, already identified internally, but by itself it is not separator-level here. Written out mathematically, it is still mostly “better belief auxiliary targets,” not a capability-changing compute/teacher mechanism. ([GitHub][4])
    * **Hand-EV realism rewrite:** likely strong, already high-priority, but it is mainline closure rather than a fresh breakthrough candidate. It improves an existing Hydra bet; it does not solve the partial-observability teacher-misalignment problem. ([GitHub][5])
    * **Stronger endgame exactification:** promising and probably worth doing, but too slice-specific for this invention pass. It is high leverage in late-game states, not the best first separator for Hydra’s current mainline bottleneck. ([GitHub][4])
    * **Confidence-gated safe exploitation residual:** attractive because it is cheap, but once written out it is mostly a bounded root residual on top of an archetype posterior. Without better posterior quality, it is incremental and brittle. ([GitHub][4])
    * **Full public-history subgame decomposition / explicit history filtering:** rejected because it fights Hydra’s selective-search posture and because history filtering / subgame-root construction can become intractable outside favorable domains. It is too diffuse and too likely to turn into broad-search identity creep. ([arXiv][12])

  [1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md"
  [2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md"
  [3]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md"
  [4]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md"
  [5]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md"
  [6]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md"
  [7]: https://arxiv.org/abs/2602.20574 "https://arxiv.org/abs/2602.20574"
  [8]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md"
  [9]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md"
  [10]: https://arxiv.org/abs/2410.00620 "https://arxiv.org/abs/2410.00620"
  [11]: https://arxiv.org/abs/2212.06027 "https://arxiv.org/abs/2212.06027"
  [12]: https://arxiv.org/abs/2311.14651 "https://arxiv.org/abs/2311.14651"
  ]]>
  </answer_text>
  </answer_section>
</combined_run_record>
