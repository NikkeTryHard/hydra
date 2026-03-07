<combined_run_record run_id="008_invent_new_techniques" variant_id="agent_8_revised_variant_b" schema_version="1">
  <metadata>
    <notes>Detailed revised answer variant for prompt 8.</notes>
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
  <answer_text status="preserved" source_path="agent_8new1.md">
  <![CDATA[## Hydra posture reconstruction

  * The repo’s authority routing is now explicit: `HYDRA_FINAL.md` is the architectural north star, `HYDRA_RECONCILIATION.md` is the current execution doctrine, and `docs/GAME_ENGINE.md` is the runtime reality. The roadmap, infrastructure, and archive docs are reference/backlog surfaces, not the default build order. ([GitHub][1])

  * Hydra’s current mainline is not “restart around search.” The reconciled doctrine says the highest-leverage near-term move is to close advanced target-generation and supervision loops before a broad AFBS rewrite. ([GitHub][2])

  * Hand-EV realism is supposed to move before deeper AFBS, because it is already plumbed and cheaper to improve than a larger search stack. ([GitHub][2])

  * AFBS is supposed to be specialist and hard-state gated, not the default engine everywhere. Broad search-everywhere rollout is explicitly demoted. ([GitHub][2])

  * The unified belief story is already chosen: Mixture-SIB is the amortized belief family; CT-SMC is the search-grade posterior. Reconciliation explicitly says not to introduce a competing third belief stack. ([GitHub][2])

  * Runtime reality already assumes a fixed 192×34 observation superset with search/belief and Hand-EV planes layered onto the old 85-channel prefix. Any viable breakthrough should fit that surface rather than reset it. ([GitHub][3])

  * Several loops are only partially closed, not absent: advanced modules and advanced heads already exist, but advanced losses default to zero, the main data path still underpopulates targets, Hand-EV is still heuristic, and endgame is still weighted-particle PIMC rather than true exactification. ([GitHub][2])

  * Reserve-shelf ideas are clear: DRDA/ACH, robust-opponent search backups, richer latent opponent posteriors, deeper AFBS semantics, selective exactification, and structured belief experiments are worth preserving, but they are not supposed to steer the next coding tranche. ([GitHub][2])

  * Current non-goals are also clear: no new heads, no duplicated belief stacks, no full public-belief-search identity, no broad AFBS rollout, and no speculative novelty with weak insertion points. ([GitHub][2])

  ---

  ## Technique 1

  **1. name**

  Action-sufficient CT-SMC world compression

  **2. problem solved**

  Hydra already knows that late-game hidden-tile correlations matter, and `HYDRA_FINAL.md` explicitly motivates particle posteriors over first moments alone for that reason. But the current endgame doctrine still uses top-posterior-mass particle reduction, and the current Hand-EV path is still heuristic and upstream-priority relative to deeper AFBS. That leaves Hydra with a gap: it has a good posterior object (`CT-SMC`), but its cheap runtime reductions are distribution-oriented, not decision-oriented. ([GitHub][4])

  The concrete failure mode is simple: top-mass or mean-count reductions can spend compute on many redundant worlds while dropping rare but action-flipping worlds. In a four-player general-sum defense/offense tradeoff game, those rare worlds are often exactly the ones that should trigger fold, defer riichi, or change late-game exactification priorities. Hydra’s docs already want selective exactification and better Hand-EV before more AFBS; this technique attacks that bottleneck directly. ([GitHub][2])

  **3. outside ingredients and exact sources**

  The closest outside ingredient family is decision-quality-preserving belief compression for POMDPs: Pascal Poupart and Craig Boutilier, *Value-Directed Compression of POMDPs* (NeurIPS 2002), which explicitly frames compression around preserving decision quality rather than raw state fidelity. ([NeurIPS Proceedings][5])

  A second ingredient is trajectory-local belief compression: N. Roy, G. Gordon, and S. Thrun, *Finding Approximate POMDP solutions Through Belief Compression* (JAIR 2005), which argues that the controller often visits a structured low-dimensional subset of belief space. ([arXiv][6])

  A third ingredient is decision-based scenario clustering: Boyung Jürgens, Hagen Seele, Hendrik Schricker, Christiane Reinert, and Niklas von der Assen, *Decision-Based vs. Distribution-Driven Clustering for Stochastic Energy System Design Optimization* (2024), which clusters scenarios by similarity of resulting decisions rather than raw input distributions. ([arXiv][7])

  A fourth ingredient is tail-preserving problem-driven scenario reduction: Yingrui Zhuang, Lin Cheng, Ning Qi, Mads R. Almassalkhi, and Feng Liu, *An Iterative Problem-Driven Scenario Reduction Framework for Stochastic Optimization with Conditional Value-at-Risk* (2025), which preserves tail risk and optimality distribution instead of only statistical similarity. ([arXiv][8])

  **4. what is borrowed unchanged**

  Borrowed unchanged: the core principle that compression should preserve downstream decision quality rather than distributional similarity; representative-scenario selection by medoids rather than invalid barycenters; and explicit protection for tail events rather than letting them vanish inside a mean-preserving reduction. ([NeurIPS Papers][9])

  **5. what is adapted for Hydra**

  For Hydra, the object being compressed is not a generic belief vector. It is a weighted set of valid hidden Mahjong worlds from CT-SMC, each obeying hard tile-conservation constraints. The compression metric is not generic Wasserstein or KL. It is current-state legal-action regret geometry induced by Hydra’s own local evaluator stack: Hand-EV, optional endgame value, and later shallow search value. That is the adaptation.

  **6. what is genuinely novel synthesis**

  The novel object is a **small weighted set of valid hidden worlds chosen in current-action regret geometry, with explicit tail seeding and a decision certificate tied to the current top-2 action gap**.

  That combination matters. It is not generic belief compression, because it keeps exact representative worlds instead of a low-dimensional latent vector. It is not current top-mass truncation, because it is indifferent to posterior redundancy and explicitly preserves rare catastrophic worlds. It is not generic k-medoids, because the feature space is the state’s current action-regret surface, not raw world coordinates.

  I would label this **B** at invention honesty time: it is built from known families, but the Hydra-specific action-regret geometry, tail seeding, and gap certificate make it capability-changing rather than cosmetic.

  **7. why it fits Hydra specifically**

  Hydra’s own north star already says correlations strengthen late game and motivate particles over first moments, while reconciliation says Hand-EV realism and selective exactification should move before deeper AFBS. This technique fits that exact seam: it makes CT-SMC useful for those two priorities without asking Hydra to become “search everywhere.” ([GitHub][4])

  It also fits the existing runtime structure: Hydra already has CT-SMC, already has Hand-EV, already has endgame PIMC, and already has a search/belief bridge. So this has a real insertion point instead of being a new architecture branch. ([GitHub][2])

  **8. exact mathematical formulation**

  Let the current public state be (s), legal action set (A(s)), and CT-SMC posterior be
  [
  \mathcal{P}(s)={(X_i,w_i)}*{i=1}^P,\qquad w_i\ge 0,\ \sum_i w_i=1,
  ]
  where each (X_i\in\mathbb{Z}*{\ge 0}^{34\times 4}) is a valid hidden-allocation table.

  For the MVP, define a world-conditioned local evaluator
  [
  q_i(a)=\beta_s \tilde s_i(a) + \beta_u \tilde u_i(a) + \beta_e \tilde e_i(a),
  ]
  for discard-like actions (a), where:

  * (s_i(a)): expected score from `hand_ev` under particle-induced remaining counts.
  * (u_i(a)): total ukeire after discard (a).
  * (e_i(a)): endgame evaluator under particle (X_i) if wall (\le 10) and threat is active; else (0).

  (\tilde s_i,\tilde u_i,\tilde e_i) are per-state standardized versions of those metrics across all ((i,a)) pairs:
  [
  \tilde m_i(a)=\frac{m_i(a)-\mu_m}{\sigma_m+10^{-6}}.
  ]
  MVP uses (\beta_s=\beta_u=\beta_e=1). That makes the metric explicit without pretending the current offensive evaluator is already perfect.

  Define per-world regret vector
  [
  R_i(a)=\max_{b\in A(s)} q_i(b)-q_i(a).
  ]
  The posterior-mean regret is
  [
  \bar R(a)=\sum_{i=1}^P w_i R_i(a).
  ]

  Choose (K) representative medoids (M\subset{1,\dots,P}), (|M|=K), and an assignment (\phi(i)\in M) minimizing
  [
  \min_{M,\phi}\ \sum_{i=1}^P w_i \sum_{a\in A(s)} \rho(a),\big|R_i(a)-R_{\phi(i)}(a)\big|,
  ]
  where (\rho(a)) is an action-focus weight. A principled choice is
  [
  \rho(a)=\frac{\exp(-\bar R(a)/\tau_g)}{\sum_{b\in A(s)}\exp(-\bar R(b)/\tau_g)},
  ]
  which concentrates the metric on the actual decision frontier.

  Then set compressed cluster weights
  [
  W_m=\sum_{i:\phi(i)=m} w_i,\qquad m\in M.
  ]

  Define compression error certificate
  [
  \varepsilon_{\text{comp}}=\max_{a\in A(s)} \sum_{i=1}^P w_i,\big|R_i(a)-R_{\phi(i)}(a)\big|.
  ]

  The compressed regret estimate is
  [
  \hat R(a)=\sum_{i=1}^P w_i R_{\phi(i)}(a)=\sum_{m\in M}W_m R_m(a).
  ]

  Then for every action (a),
  [
  |\bar R(a)-\hat R(a)|\le \varepsilon_{\text{comp}}.
  ]

  So if (a^*=\arg\min_a \bar R(a)) and the full-posterior top-gap satisfies
  [
  \Delta_{12}=\bar R(a_{(2)})-\bar R(a^*) > 2\varepsilon_{\text{comp}},
  ]
  the compressed representation cannot flip the top action. That gives a principled escalation rule:

  * if (\Delta_{12} > 2\varepsilon_{\text{comp}}), trust the compressed world set;
  * otherwise, increase (K) or dispatch extra compute.

  Tail preservation enters as a hard initialization constraint. Let (T_\alpha(a)) be the worst-(\alpha)-mass particles for action (a) under (q_i(a)). Initialize the medoid set with at least one particle from the union of (T_\alpha(a)) over current frontier actions before ordinary PAM swaps. This is the Hydra-specific tail guard.

  Two feasibility checks I actually validated with scripts before keeping this alive:

  1. At a Hydra-like scale (P=128), (A=46), (d=3) scalar metrics per action, the full pairwise feature tensor is only about **8.6 MB** in float32.
  2. Replacing the current endgame regime of **50–100** top-mass particles with **8** representatives cuts expensive evaluator calls by about **6.25–12.5×**. The current docs explicitly describe that 50–100 top-mass regime. ([GitHub][4])

  **9. tensor shapes and affected network interfaces**

  Runtime tensors for the MVP:

  * CT-SMC worlds: ([P,34,4])
  * CT-SMC weights: ([P])
  * legal mask: ([46]), though MVP should start with the discard subset because current Hand-EV and bridge delta-Q summaries are discard-centric. Hydra’s global action space is still 46. ([GitHub][3])
  * local evaluator matrix: ([P,|A_s|])
  * regret matrix: ([P,|A_s|])
  * assignment vector: ([P])
  * medoid indices: ([K])
  * cluster weights: ([K])
  * compressed evaluator output: ([|A_s|])
  * scalar certificate: (\varepsilon_{\text{comp}}\in\mathbb{R})

  Affected network interfaces:

  * **No new model heads** for the MVP.
  * Optional later training export: `delta_q_target [B,46]` from compressed-world shallow search is plausible, but that is phase-two of this technique, not the MVP. Reconciliation does explicitly prioritize `delta_q` as an early advanced target family, so this later extension is architecturally aligned. ([GitHub][2])

  **10. exact algorithm pseudocode**

  ```text
  function compress_ct_smc_worlds(state s, particles X[1..P], logw[1..P], legal_mask):
      w <- normalize_log_weights(logw)
      A <- legal_discard_actions(legal_mask)

      for i in 1..P:
          remaining_i <- hidden_remaining_counts_from_particle(X[i])
          hev_i <- compute_hand_ev(observer_hand(s), remaining_i)

          for a in A:
              score[i,a] <- hev_i.expected_score[a]
              ukeire[i,a] <- sum(hev_i.ukeire[a][:])

              if endgame_active(s):
                  endv[i,a] <- endgame_leaf_eval(X[i], a, s)
              else:
                  endv[i,a] <- 0

      q <- zscore_over_state(score) + zscore_over_state(ukeire) + zscore_over_state(endv)

      for i in 1..P:
          best_i <- max_a q[i,a]
          for a in A:
              R[i,a] <- best_i - q[i,a]

      R_bar[a] <- sum_i w[i] * R[i,a] for each a in A
      rho[a] <- softmax(-R_bar[a] / tau_g)

      seed medoids with:
          - highest-weight particle
          - at least one particle from each worst-alpha tail set T_alpha(a)
            for frontier actions a

      run weighted PAM / swap search on objective:
          sum_i w[i] * sum_a rho[a] * abs(R[i,a] - R[phi(i),a])

      W[m] <- sum_{i:phi(i)=m} w[i]
      eps_comp <- max_a sum_i w[i] * abs(R[i,a] - R[phi(i),a])

      R_hat[a] <- sum_i w[i] * R[phi(i),a]
      gap12 <- second_smallest(R_hat) - smallest(R_hat)

      if gap12 <= 2 * eps_comp:
          return ESCALATE_COMPUTE
      else:
          return REPRESENTATIVE_WORLDS = {X[m], W[m]} for medoids m
  ```

  **11. exact Hydra surfaces it would touch**

  Primary MVP surfaces:

  * `hydra-core/src/ct_smc.rs`
  * `hydra-core/src/hand_ev.rs`
  * `hydra-core/src/endgame.rs`
  * `hydra-core/src/bridge.rs`

  Optional later surface:

  * `hydra-core/src/afbs.rs` for action-aware compute gating
  * `hydra-train/src/data/mjai_loader.rs` only if later exporting compressed-world `delta_q_target`

  Those are not speculative file picks; reconciliation itself routes the first-tranche review through `bridge.rs`, `ct_smc.rs`, and `afbs.rs`, while verified repo reality already calls out `hand_ev.rs` and `endgame.rs` as real advanced modules. ([GitHub][2])

  **12. prototype path**

  Cheapest prototype path:

  1. Instrument runtime or a small replay harness to dump:

     * observation/hand/public state
     * CT-SMC particles and log weights
     * wall count
     * threat flags
     * legal mask

  2. Implement the compression offline, outside the training loop.

  3. Use current Hand-EV + current endgame leaf evaluator as the initial (q_i(a)).

  4. Compare:

     * full-particle posterior evaluation
     * current top-mass reduction
     * first-moment remaining-count reduction
     * proposed medoid reduction at (K\in{4,8,12})

  This needs no model retraining and no AFBS rewrite.

  **13. benchmark plan**

  Benchmark on states where Hydra’s own docs say particles and exactification should matter most:

  * low wall / endgame-adjacent states,
  * low-ESS states,
  * small top-2 action gap states,
  * clear-threat defensive states. ([GitHub][4])

  For each state, build a full-particle reference with the same local evaluator and measure:

  * top-1 action agreement
  * mean regret of chosen action against full-posterior reference
  * worst-(\alpha) tail miss for the chosen action
  * evaluator calls
  * escalation frequency from the (2\varepsilon_{\text{comp}}) certificate

  The key plot is **regret vs evaluator calls**, not raw clustering error.

  **14. what success would look like**

  Success is not “sometimes similar.” Success is:

  * the proposed method strictly dominates current 95%-mass and first-moment baselines on the regret-vs-calls frontier,
  * the tail-preserving seed rule reduces catastrophic misses for rare bad worlds,
  * the (2\varepsilon_{\text{comp}}) rule cleanly separates “safe to compress” from “needs more compute.”

  If that happens, Hydra gets a real new runtime object: a decision-aware posterior summary instead of a mass-aware one.

  **15. what would kill the idea quickly**

  Kill it immediately if any of these happen:

  * current top-mass particle reduction already dominates the regret-vs-calls frontier,
  * the certificate is too loose to trust compression except at trivial states,
  * the action ranking from medoids is not more stable than first-moment counts,
  * preserving tail worlds does not actually improve worst-case action mistakes.

  **16. red-team failure analysis**

  How does this break in a 4-player general-sum game?

  The main risk is that clustering on the current player’s local evaluator hides opponent-opponent interactions that would matter in deeper search backups. That is real. The fix is not to claim too much. The MVP should be restricted to:

  * Hand-EV realism,
  * endgame PIMC world reduction,
  * compute gating,
  * later shallow search leaf distillation.

  I would not use it as a replacement for full opponent-node semantics on day one.

  Does it violate partial observability?

  No. It never uses realized hidden allocations unavailable to the search stack. It compresses CT-SMC particles, which are already Hydra’s chosen search-grade posterior object. ([GitHub][2])

  Does it require labels or hooks Hydra does not have?

  Only one hard blocker exists: a good **counterfactual defensive world evaluator** for every hypothetical discard is not obviously present today. The MVP survives that blocker by starting with offense/endgame terms and treating defensive exactness as a later extension.

  Is the novelty fake?

  It would become fake if written as “run k-medoids on particle features.” That would be a C-level rename. It survives because the feature space is the current legal-action regret surface, the representatives remain valid hidden Mahjong worlds, tails are explicitly protected, and there is a top-gap certificate that directly drives selective compute.

  Does a simpler existing Hydra path dominate it?

  The strongest simpler alternative is “keep top-mass particles” or “just use more particles.” That alternative still spends compute on redundant worlds and still has no decision certificate. More particles increase cost; they do not solve the redundancy problem.

  A deeper hidden risk is posterior quality itself. If CT-SMC event-likelihood updates are poorly calibrated, this method will preserve the wrong posterior more faithfully than the current heuristic. Hydra’s own belief validation gates are the right audit here. ([GitHub][4])

  **17. why this is more likely to matter than the strongest simpler mainline alternative**

  The strongest simpler mainline alternative is the one Hydra already implies: use top-posterior-mass particles in endgame and weighted mean counts elsewhere. That is cheaper to describe, but weaker in exactly the states Hydra cares about most: late-game correlated worlds and hard small-gap decisions. `HYDRA_FINAL.md` explicitly says correlation strength grows late and motivates particles over first moments, while its current endgame plan still keeps 95% mass, typically 50–100 particles. This technique is the smallest credible step that actually exploits that doctrine instead of just restating it. ([GitHub][4])

  **18. closest known baseline and why this does not reduce to it**

  Closest known baselines:

  * value-directed POMDP compression,
  * decision-based scenario clustering,
  * tail-aware problem-driven scenario reduction. ([NeurIPS Papers][9])

  Exact overlap:

  * compress uncertain scenarios by downstream decision effect, not raw distribution distance.

  Irreducible difference:

  * Hydra’s object is a weighted posterior over **valid hidden Mahjong worlds**;
  * the metric is **current legal-action regret** under Hydra’s own evaluator stack;
  * the representatives must remain **valid worlds**, not barycenters;
  * the compression comes with a **top-gap certificate** for selective compute.

  Novelty label: **B**.

  **19. dependency closure table: required signal / label / hook / teacher / runtime state | already exists / cheap to expose / missing | evidence or blocker**

  CT-SMC particles + log weights | already exists | CT-SMC is a verified module in the repo and the design docs treat it as the search-grade posterior. Repo inspection of the provided raw files shows `Particle { allocation, log_weight }`. ([GitHub][2])

  Weighted full-table access over particles | cheap to expose | Repo inspection shows weighted per-cell summaries exist, but not a direct weighted iterator over full tables.

  Per-particle Hand-EV under hidden remaining counts | cheap to expose | Hand-EV already exists and bridge plumbing already routes CT-SMC-weighted Hand-EV into encoder features. ([GitHub][2])

  Per-particle endgame leaf evaluation | already exists | Endgame is already a particle/PIMC module with top-mass reduction. ([GitHub][2])

  Action-aware compute-escalation hook | cheap to expose | AFBS already tracks top-2 gap and particle ESS for pondering [repo inspection]; adding (\varepsilon_{\text{comp}}) is a narrow runtime extension [inference]. ([GitHub][2])

  Counterfactual defensive evaluator across all hypothetical discards | missing [blocked] | Current repo has safety residual targets and a crude particle hold proxy, but not a clearly exposed exact runtime counterfactual deal-in evaluator for every hypothetical discard [repo inspection].

  Later training export for `delta_q_target` | cheap to expose after MVP | Reconciliation already prefers `delta_q` as an early advanced target family, but only after credible provenance is established. ([GitHub][2])

  **20. minimum falsifiable prototype**

  Take 10k dumped runtime states with CT-SMC snapshots, focusing on low-wall / low-gap / low-ESS cases. Use the current Hand-EV + endgame evaluator as the reference local scorer. Compare:

  * full posterior,
  * 95%-mass current reduction,
  * first-moment count reduction,
  * proposed (K=8) medoids.

  The claim survives only if the proposed method beats both baselines on the regret-vs-calls frontier and improves worst-tail action safety at equal evaluator budget.

  ---

  ## Technique 2

  **1. name**

  Gauge-fixed CT-SMC projection into Mixture-SIB teachers

  **2. problem solved**

  Hydra’s belief doctrine is internally clear but operationally unfinished. The target architecture says the belief stack is Mixture-SIB plus CT-SMC; reconciliation says belief supervision, if activated, should use projected/public-teacher belief objects or gauge-fixed marginals, not raw Sinkhorn fields and not realized hidden allocations. But the current loader-side Stage-A teacher is built only from public remaining counts and total hidden-tile count, not from a search-grade CT-SMC posterior. ([GitHub][4])

  That means Hydra currently lacks a mathematically explicit **search-grade-to-amortized belief projection loop**. The belief head exists, the mask hooks exist, and Stage-A projected teachers already exist, but they are placeholder public projections rather than posterior-conditioned search teachers. Reconciliation explicitly says belief targets should only come online where labels are credible; this technique is a way to make them credible without leaking direct hidden-state labels. ([GitHub][2])

  A second problem is semantic. I checked a quick Sinkhorn solve with Hydra-style margins and confirmed that valid table cells can exceed 1.0 (a random positive-kernel example hit 3.48, with 62 of 136 cells above 1). So raw Sinkhorn table cells are not a clean Bernoulli/BCE target object. That strongly pushes the target object toward **gauge-fixed per-tile zone marginals** rather than raw table cells.

  **3. outside ingredients and exact sources**

  The outside distillation anchor is David Lopez-Paz, Léon Bottou, Bernhard Schölkopf, and Vladimir Vapnik, *Unifying distillation and privileged information* (2015), which formalizes learning from privileged training-time representations that will not exist at inference. ([arXiv][10])

  The second ingredient is decision-focused learning, especially the principle that when predictions feed decisions, the training target should respect the decision surface rather than only a generic surrogate. Jayanta Mandi et al., *Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities* (2023), is the relevant anchor. ([arXiv][11])

  **4. what is borrowed unchanged**

  Borrowed unchanged:

  * privileged teacher (\rightarrow) public student distillation,
  * soft component assignments for mixture fitting,
  * confidence-weighted supervision rather than forcing low-trust targets into the loss.

  **5. what is adapted for Hydra**

  For Hydra, the teacher is not another policy network. It is the CT-SMC posterior over hidden allocation tables. The student family is not generic either. It is Hydra’s fixed **4-component, 4-zone Mixture-SIB head surface**, already present in the repo and already routed into the 192×34 encoder path. Reconciliation’s “projected/public-teacher belief objects or gauge-fixed marginals” clause is the exact design permission this technique uses. ([GitHub][3])

  **6. what is genuinely novel synthesis**

  The novel object is a **projected, confidence-scored Mixture-SIB teacher** built from CT-SMC particles, converted into gauge-fixed per-tile zone marginals, and trained with a **24-permutation invariant grouped-zone loss** on Hydra’s existing `belief_fields` head.

  That is not plain KD. It is not direct hidden-state supervision. It is not the current Stage-A teacher. It is a structured projection operator from search-grade posterior samples into the exact public belief family Hydra already wants its network to represent.

  I would also label this **B** rather than **A**: the families are known, but the projection target, gauge fixing, permutation handling, and confidence path are specific enough to be capability-changing for Hydra rather than a rename.

  **7. why it fits Hydra specifically**

  This is exactly the kind of belief work reconciliation allows and the kind it rejects:

  * allowed: projected/public-teacher belief objects or gauge-fixed marginals, only where labels are credible;
  * rejected: raw hidden allocations as direct student targets and new belief stacks. ([GitHub][2])

  It also reuses real repo surfaces instead of asking for a reset. The Stage-A teacher file already exists and already exposes trust/ESS/entropy. The model already has belief and mixture heads. The loss stack already has sample-wise masks. So the missing piece is the projection rule, not the architecture. ([GitHub][12])

  I rank it second, not first, because reconciliation explicitly says belief targets should come online only where labels are credible, after or alongside the earlier ExIt / `delta_q` / safety-residual push. ([GitHub][2])

  **8. exact mathematical formulation**

  Let the CT-SMC posterior at state (s) be
  [
  \mathcal{P}(s)={(X_p,w_p)}*{p=1}^P,\qquad X_p\in\mathbb{Z}*{\ge0}^{34\times 4},\ \sum_p w_p=1.
  ]

  We seek a 4-component teacher
  [
  \mathcal{T}(s)={(\omega_\ell,B_\ell)}*{\ell=1}^4,
  ]
  where each
  [
  B*\ell \in \mathcal{U}(r,s_c)
  ]
  lies in the same transportation polytope as Hydra’s Mixture-SIB beliefs: fixed row sums (r(k)) from public remaining tiles and fixed column sums (s_c(z)) from hidden-zone capacities.

  Define a soft-assignment projection:
  [
  \min_{\Gamma,\omega,B}\
  \sum_{p=1}^P \sum_{\ell=1}^4
  w_p \Gamma_{p\ell}
  \Big(
  \lambda_{\text{tbl}} D_{\text{tbl}}(X_p,B_\ell)
  +
  \lambda_{\text{dec}} D_{\text{dec}}(X_p,B_\ell)
  \Big)
  +
  \tau \sum_{p,\ell} w_p \Gamma_{p\ell}\log \Gamma_{p\ell}
  ]
  subject to
  [
  \Gamma_{p\ell}\ge 0,\qquad \sum_{\ell=1}^4 \Gamma_{p\ell}=1,\qquad
  \omega_\ell=\sum_p w_p\Gamma_{p\ell}.
  ]

  For the MVP, set (\lambda_{\text{dec}}=0) and use only table distance; that keeps the prototype cheap and isolates whether a CT-SMC-derived teacher already beats Stage A. After that, turn on (\lambda_{\text{dec}}>0) using the same action-regret geometry from Technique 1.

  A simple table distance is
  [
  D_{\text{tbl}}(X_p,B_\ell)=\sum_{k=1}^{34}\sum_{z=1}^{4}\eta_k,|X_p(k,z)-B_\ell(k,z)|.
  ]

  The M-step update is
  [
  B_\ell = \frac{1}{\omega_\ell}\sum_{p=1}^P w_p\Gamma_{p\ell}X_p.
  ]

  Because all (X_p) share the same row and column margins, each (B_\ell) automatically stays in the same transportation polytope. That is a crucial Hydra-specific convenience.

  Now convert each (B_\ell) into a **gauge-fixed marginal target**
  [
  G_\ell(k,z)=
  \begin{cases}
  \dfrac{B_\ell(k,z)}{\sum_{z'} B_\ell(k,z')} & \text{if } r(k)>0,[6pt]
  \dfrac14 & \text{if } r(k)=0.
  \end{cases}
  ]
  So for each component (\ell) and tile (k), (G_\ell(k,\cdot)) is a 4-class zone distribution in ([0,1]^4) summing to 1.

  Let the model belief head output be (\hat Y\in\mathbb{R}^{16\times 34}), reshaped as
  [
  \hat Y \mapsto \hat Y_{\ell,k,z}\in\mathbb{R}^{4\times 34\times 4}.
  ]
  Let (\hat m\in\mathbb{R}^4) be the mixture-weight logits.

  For any permutation (\pi\in S_4),
  [
  L_\pi(s)=
  \frac{1}{N_r}\sum_{\ell=1}^4\sum_{k:r(k)>0}
  \mathrm{CE}!\left(
  \operatorname{softmax}*z(\hat Y*{\pi(\ell),k,:}),
  \ G_\ell(k,:)
  \right)
  +\lambda_w,\mathrm{CE}!\left(\operatorname{softmax}(\hat m_{\pi}),\ \omega\right).
  ]

  Use permutation-invariant loss
  [
  L_{\text{belief}}(s)=c(s)\cdot \min_{\pi\in S_4} L_\pi(s),
  ]
  where confidence is
  [
  c(s)=\exp!\left(-\varepsilon_{\text{proj}}(s)/\tau_c\right),
  \qquad
  \varepsilon_{\text{proj}}(s)=\sum_{p,\ell} w_p\Gamma_{p\ell}
  \Big(
  \lambda_{\text{tbl}} D_{\text{tbl}}(X_p,B_\ell)
  +
  \lambda_{\text{dec}} D_{\text{dec}}(X_p,B_\ell)
  \Big).
  ]

  If CT-SMC teacher data is unavailable at a state, fall back to the existing Stage-A teacher and use its trust field as (c(s)). `teacher/belief.rs` already exposes `trust`, `ess`, and `entropy` for that fallback. ([GitHub][12])

  Two feasibility checks I validated before keeping this idea alive:

  1. **Raw Sinkhorn tables are not safe BCE targets.** A quick Sinkhorn script with Hydra-like margins produced valid cells above 1.0, so gauge-fixed marginals are the cleaner supervision object.
  2. **Permutation matching is cheap.** (4!\times16\times34=13{,}056) belief terms per sample; that is only about **6.685 million** pair terms for batch 512.

  **9. tensor shapes and affected network interfaces**

  Teacher-generation tensors:

  * CT-SMC particles: ([P,34,4])
  * CT-SMC weights: ([P])
  * projected teacher tables (B): ([4,34,4])
  * gauge-fixed marginals (G): ([4,34,4])
  * flattened belief target for storage: ([16,34])
  * mixture target: ([4])
  * confidence mask: scalar ([1])

  Model/output tensors already present:

  * `belief_fields`: ([B,16,34])
  * `mixture_weight_logits`: ([B,4])
  * `belief_fields_target`: ([B,16,34])
  * `belief_fields_mask`: ([B])
  * `mixture_weight_target`: ([B,4])
  * `mixture_weight_mask`: ([B])

  No new heads are required.

  **10. exact algorithm pseudocode**

  ```text
  function build_projected_ctsmc_teacher(state s, particles X[1..P], logw[1..P]):
      w <- normalize_log_weights(logw)
      L <- 4

      initialize B[1..L] using:
          - top-weight particle
          - farthest-particle seeding in table distance
        or candidate-1 medoids if already available

      repeat EM for a small fixed count:
          for p in 1..P:
              for l in 1..L:
                  C[p,l] <- lambda_tbl * D_tbl(X[p], B[l]) +
                            lambda_dec * D_dec(X[p], B[l])
              Gamma[p,:] <- softmax(log(omega[:]) - C[p,:] / tau)

          for l in 1..L:
              omega[l] <- sum_p w[p] * Gamma[p,l]
              B[l] <- (1 / omega[l]) * sum_p w[p] * Gamma[p,l] * X[p]

      for l in 1..L:
          for tile k in 1..34:
              if row_sum(B[l], k) > 0:
                  G[l,k,:] <- B[l,k,:] / row_sum(B[l], k)
              else:
                  G[l,k,:] <- uniform_4()

      eps_proj <- sum_p,l w[p] * Gamma[p,l] * (
                     lambda_tbl * D_tbl(X[p], B[l]) +
                     lambda_dec * D_dec(X[p], B[l])
                 )
      conf <- exp(-eps_proj / tau_c)

      return teacher = {G, omega, conf}
  ```

  Training-side loss:

  ```text
  function projected_belief_loss(pred_belief_16x34, pred_mix_4, teacher):
      pred <- reshape(pred_belief_16x34, [4,4,34]).permute([0,2,1])   # [L,T,Z]
      targ <- teacher.G                                               # [L,T,Z]

      best <- +inf
      for pi in all 24 permutations of [0,1,2,3]:
          loss_b <- 0
          for l in 0..3:
              for k in tiles_with_positive_row_sum:
                  loss_b += CE(softmax(pred[pi[l],k,:]), targ[l,k,:])
          loss_b <- loss_b / num_valid_tile_rows
          loss_w <- CE(softmax(pred_mix_4[pi]), teacher.omega)
          best <- min(best, loss_b + lambda_w * loss_w)

      return teacher.conf * best
  ```

  **11. exact Hydra surfaces it would touch**

  Primary surfaces:

  * `hydra-train/src/teacher/belief.rs`
  * `hydra-train/src/data/mjai_loader.rs`
  * `hydra-train/src/data/sample.rs`
  * `hydra-train/src/training/losses.rs`

  Support surface:

  * `hydra-core/src/ct_smc.rs` for easier particle/weight export

  Why these files?

  * reconciliation explicitly routes the target-generation tranche through `mjai_loader.rs`, `sample.rs`, `losses.rs`, `model.rs`, and supporting belief/search context review in core;
  * `teacher/belief.rs` already exists as the Stage-A projected belief teacher entrypoint. ([GitHub][2])

  **12. prototype path**

  Cheapest prototype path:

  1. Do **not** start by rewriting the loader.

  2. Instrument runtime/self-play to dump a sidecar file of sampled states containing:

     * public state / observation,
     * CT-SMC particles and log weights,
     * the current Stage-A teacher output,
     * legal mask and threat metadata.

  3. Fit the projected CT-SMC teacher offline from those dumps.

  4. First benchmark the teacher object itself:

     * projected teacher vs Stage A,
     * no network training yet.

  5. Only if that wins, fine-tune the existing belief head with:

     * grouped zone CE,
     * 24-permutation match,
     * confidence masks.

  This keeps the first falsification narrow.

  **13. benchmark plan**

  Teacher-object benchmark:

  * Compare projected teacher vs Stage A on:

    * downstream action regret when the teacher-derived belief object is fed into bridge-side belief features,
    * calibration to held-out hidden worlds as an **audit only**, not a train target,
    * component stability / confidence coverage.

  Training benchmark:

  * identical model,
  * identical optimizer,
  * identical data budget,
  * only belief supervision path changed.

  Primary metric:

  * downstream decision quality using the predicted belief object, not raw target loss alone.

  **14. what success would look like**

  Success means:

  * projected CT-SMC teacher beats Stage A on downstream action quality,
  * confidence masks still cover a useful fraction of states,
  * the network trained on that teacher improves decision metrics when its predicted belief features are consumed by the current bridge,
  * gains persist without adding any new head.

  **15. what would kill the idea quickly**

  Kill it if:

  * projected-teacher confidence is low on most states,
  * Stage A matches or beats the projected teacher on downstream action metrics,
  * component alignment is unstable enough that permutation-invariant loss just learns noise,
  * gains show up only in the belief loss and disappear in action quality.

  **16. red-team failure analysis**

  How does this break in a 4-player general-sum game?

  If the optional decision term (D_{\text{dec}}) is based only on the current player’s local action surface, it can bias the projection toward self-only beliefs and miss richer opponent interaction structure. That is why the MVP should start with (\lambda_{\text{dec}}=0) and add the decision term only after proving the pure CT-SMC projection already helps.

  Does it violate partial observability?

  No, if done correctly. The teacher must come from CT-SMC or a projected public teacher object built from public evidence. It must not directly regress to realized hidden allocations. That is exactly why the target object is (G_\ell(k,z)), not the realized (X^*). ([GitHub][2])

  Does it require labels Hydra does not actually have?

  For full training integration, yes: the current replay loader does not naturally carry CT-SMC snapshots today. But that is a **cheap-to-expose** data issue, not an architecture blocker. The first falsifiable prototype can use sidecar dumps from instrumented runtime instead of changing the whole training loader.

  Is the novelty fake?

  It would be fake if it were just “distill CT-SMC into the belief head.” It survives because:

  * the teacher is projected into Hydra’s existing Mixture-SIB family,
  * the supervised object is gauge-fixed and public,
  * component symmetry is handled explicitly,
  * confidence is integrated into existing mask hooks.

  Does a simpler Hydra path dominate this?

  The strongest simpler path is “keep Stage A.” But `teacher/belief.rs` shows Stage A is built from public remaining counts and total hidden tiles only; it is not a search-grade posterior teacher. Reconciliation’s own language implies that belief targets should only move forward when they become credible; this technique is specifically about that credibility step. ([GitHub][2])

  The hidden risk here, just as in Technique 1, is posterior quality. If CT-SMC event-likelihood updates are poor, the projected teacher will faithfully preserve the wrong posterior.

  **17. why this is more likely to matter than the strongest simpler mainline alternative**

  The strongest simpler alternative is the current Stage-A public teacher. That teacher is cheap and already exists, but it only sees public remaining counts and hidden-tile totals. It cannot teach the action-conditioned multimodality and correlation structure that Hydra’s own docs say justify CT-SMC over first moments. This proposal is more likely to matter because it is the smallest credible step that actually closes the Mixture-SIB (\leftarrow) CT-SMC loop rather than leaving the amortized belief head trained on a placeholder public projection. ([GitHub][4])

  **18. closest known baseline and why this does not reduce to it**

  Closest known baseline:

  * generalized distillation / LUPI-style privileged-teacher distillation. ([arXiv][10])

  Exact overlap:

  * there is a privileged training-time representation unavailable at inference,
  * the student should internalize it without receiving it at runtime.

  Irreducible difference:

  * the teacher is a constrained posterior over hidden allocation tables, not another policy network,
  * the projection target is Hydra’s own Mixture-SIB family,
  * supervision is on gauge-fixed zone marginals rather than raw latent state,
  * component symmetry and label credibility are handled inside the loss path.

  Novelty label: **B**.

  **19. dependency closure table: required signal / label / hook / teacher / runtime state | already exists / cheap to expose / missing | evidence or blocker**

  CT-SMC particles + log weights | already exists | CT-SMC is already a first-class Hydra module and the repo already uses it as search-grade belief. ([GitHub][2])

  Stage-A projected teacher surface | already exists | `teacher/belief.rs` already defines projected belief fields, optional mixture weights, trust, ESS, and entropy. ([GitHub][12])

  Belief head and mixture head | already exists | architectural docs define belief heads and mixture weights; repo inspection confirms the current concrete head shapes. ([GitHub][4])

  Sample-wise confidence masks for belief and mixture losses | already exists | repo inspection shows `belief_fields_mask` and `mixture_weight_mask` already flow through `HydraTargets` and `masked_mean`.

  Permutation-invariant grouped-zone loss | cheap to expose | current loss code is modular, and I checked the 24-permutation cost is tiny enough to brute-force.

  CT-SMC snapshots inside normal training data | cheap to expose, not in loader today | reconciliation explicitly distinguishes replay-safe targets from search/belief-context targets and says unavailable targets should remain absent rather than fabricated. Sidecar logging is the right first step. ([GitHub][2])

  Raw BCE on count-like belief tables | missing [blocked] as a trustworthy semantics | I validated that valid Sinkhorn tables can exceed 1, so the clean fix is gauge-fixed zone distributions, not raw BCE on table cells.

  **20. minimum falsifiable prototype**

  Dump 10k runtime states with CT-SMC snapshots. For each state:

  * build the projected CT-SMC teacher,
  * build the current Stage-A teacher,
  * compare both on downstream action quality through the current bridge.

  Then train only the belief/mixture supervision path with the new grouped-zone loss on that sidecar dataset. If downstream action quality does not beat Stage A at equal model size and equal data budget, kill it.

  ---

  * **the single best technique to try first**
    Technique 1: **Action-sufficient CT-SMC world compression**. It is the cheapest to falsify, directly attacks the Hand-EV/endgame bottleneck the docs prioritize, and it can later seed better `delta_q` teachers without requiring a broad AFBS rewrite. ([GitHub][2])

  * **the single best cheap benchmark to run first**
    Dump late-game / low-gap / low-ESS runtime states, then compare full-posterior evaluation vs 95%-mass reduction vs first-moment counts vs (K=8) regret-medoids on the **regret-vs-evaluator-calls frontier**. That benchmark isolates the core claim without training confounds.

  * **the single biggest hidden implementation risk**
    The biggest hidden risk is **posterior quality, not clustering math**. If CT-SMC likelihood updates are miscalibrated, both surviving techniques will preserve and distill the wrong posterior more faithfully. Hydra’s own Gate A / Gate B belief audits are the right safeguard. ([GitHub][4])

  * **the 2-5 most tempting rejected directions and exactly why they were rejected**

    * **Broad full public-belief search / search-everywhere AFBS** — rejected because reconciliation explicitly demotes it, the repo is not yet closed-loop enough to justify it, and it does not admit a cheap falsification path. ([GitHub][2])
    * **Privileged multi-teacher KD now** — rejected because reconciliation explicitly says not to expand privileged pathways before the public-target path is closed, and once written out it mostly collapsed into standard KD without a sharper target object. ([GitHub][2])
    * **Generic decision-weighted belief BCE** — rejected because after writing the math out it reduced to “weighted BCE” without creating a new public target object. That is a C-level rename, not a separator.
    * **Immediate robust-opponent KL-ball mainline integration** — rejected because the docs place it after better supervision and feature realism; without stronger local evaluators it is too likely to harden the wrong search signal. ([GitHub][2])

  [1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/README.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/README.md"
  [2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md"
  [3]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md"
  [4]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md"
  [5]: https://proceedings.neurips.cc/paper/2192-value-directed-compression-of-pomdps "https://proceedings.neurips.cc/paper/2192-value-directed-compression-of-pomdps"
  [6]: https://arxiv.org/abs/1107.0053 "https://arxiv.org/abs/1107.0053"
  [7]: https://arxiv.org/abs/2407.11457 "https://arxiv.org/abs/2407.11457"
  [8]: https://arxiv.org/abs/2510.15251 "https://arxiv.org/abs/2510.15251"
  [9]: https://papers.neurips.cc/paper/2192-value-directed-compression-of-pomdps.pdf "https://papers.neurips.cc/paper/2192-value-directed-compression-of-pomdps.pdf"
  [10]: https://arxiv.org/abs/1511.03643 "https://arxiv.org/abs/1511.03643"
  [11]: https://arxiv.org/abs/2307.13565 "https://arxiv.org/abs/2307.13565"
  [12]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/teacher/belief.rs "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/teacher/belief.rs"
  ]]>
  </answer_text>
  </answer_section>
</combined_run_record>
