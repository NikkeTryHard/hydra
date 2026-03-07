<combined_run_record run_id="007_cross_field_breakthrough" variant_id="agent_7_revised_variant_a" schema_version="1">
  <metadata>
    <notes>Richer revised answer variant for prompt 7.</notes>
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
  <answer_text status="preserved" source_path="agent_7new.md">
  <![CDATA[After the sequential core-doc read you asked for, then code-grounding, then outside retrieval, I end with **2 surviving candidates**. I do **not** think a third survives without quietly leaning on reserve-shelf dependencies.

  ## Hydra posture reconstruction

  * `HYDRA_RECONCILIATION.md` is the live sequencing authority. It explicitly says that if other docs conflict on sequencing, tranche priority, or active-vs-reserve status, reconciliation wins. ([GitHub][1])

  * The current blocker is **not** “Hydra lacks advanced modules.” Reconciliation says the real blocker is **partially closed loops plus doc drift**, and that the strongest immediate lever is **advanced target-generation / supervision closure**, not a giant search rewrite. ([GitHub][1])

  * The repo already contains real advanced surfaces: `ct_smc`, `sinkhorn`, `afbs`, `robust_opponent`, `hand_ev`, `endgame`, plus a live **192×34** fixed-superset encoder shared by runtime and training. ([GitHub][1])

  * The training model already exposes a large advanced head surface: policy logits, value, score pdf/cdf, opponent tenpai, GRP, next discard, danger, oracle critic, belief fields, mixture weights, opponent hand type, `delta_q`, and `safety_residual`. The oracle path is currently detached from the shared pooled representation. ([GitHub][2])

  * Reconciliation is explicit that **advanced losses exist but default advanced loss weights are zero**, and that the normal data path still underpopulates advanced targets; in particular, the first tranche should prefer **`exit_target` + `delta_q` + `safety_residual` activation first**, with belief/mixture/hand-type only when labels are credible. ([GitHub][1])

  * The project’s active doctrine is **Hand-EV realism before deeper AFBS expansion**. AFBS is valuable, but it is supposed to stay **specialist / hard-state gated**, not become the universal default engine right now. ([GitHub][1])

  * `HYDRA_FINAL.md` remains the north star: ExIt + Pondering + Search-as-Feature, Mixture-SIB + CT-SMC, AFBS, endgame exactification, robust-opponent logic, and DRDA/ACH all live there. But reconciliation demotes broad optimizer/search bets to **reserve shelf** until the target pipeline is healthier. ([GitHub][3])

  * Immediate non-goals are also explicit: **no new heads**, **no broad AFBS rewrite**, **no broad public-belief search as immediate mainline**, and **no major new optimizer family before existing surfaces are fed**. ([GitHub][1])

  * Code reality lines up with that doctrine. `bridge.rs` already has a `SearchContext` that can carry mixture beliefs, CT-SMC, AFBS, and opponent risk/stress; `mjai_loader.rs` already builds replay-safe `safety_residual` targets and stage-A belief targets; but reconciliation says production paths for `exit_target`, `delta_q_target`, and several other advanced targets are still missing in the mainline loader/train path. ([GitHub][4])

  ---

  ## Candidate 1

  ### 1. name

  **Visit-bootstrapped conservative Expert Iteration**

  ### 2. problem solved

  Hydra’s north star wants ExIt, but Hydra’s current doctrine also says AFBS is still a specialist shell, not a fully integrated public-belief runtime, and that the first tranche should activate `exit_target` / `delta_q` / `safety_residual` before broadening search. The problem is therefore not “how to search more,” but “how to extract a trustworthy improvement signal from a still-partial search teacher without poisoning the apprentice policy.” Raw `Softmax(Q/τ)` ExIt is too brittle in a 4-player general-sum hidden-information game when root children are unevenly explored or weakly supported. ([GitHub][1])

  ### 3. outside ingredients and exact sources

  The outside ingredients are:

  * **Expert Iteration**: search is a policy-improvement operator whose results are distilled into a fast apprentice. ([NeurIPS Proceedings][5])
  * **SPIBB / safe policy improvement**: stay close to the baseline policy when support is weak. ([Proceedings of Machine Learning Research][6])
  * **Q-filter from demo-RL**: use teacher pressure only when the teacher is credibly better than the current policy. ([arXiv][7])
  * **Pessimism / lower-confidence offline RL**: improvement should be based on lower-confidence estimates, not optimistic point estimates. ([arXiv][8])

  A second-order lead that mattered in the red-team pass was **HAVER**: max-selection becomes unstable when several arms are near-best, which pushes against raw argmax/softmaxing noisy search means and toward conservative, baseline-anchored projection instead. ([arXiv][9])

  ### 4. what is borrowed unchanged

  Borrowed unchanged:

  * From ExIt: “planner improves policy; apprentice imitates planner.”
  * From SPIBB/pessimism: “do not move far from the baseline when support is weak.”
  * From Q-filter: “teacher pressure should be conditional, not unconditional.”

  ### 5. what is adapted for Hydra

  The Hydra-specific adaptation is that **support** is not dataset action-count support in an MDP. It is **root-child support inside AFBS**: root child visit counts and root child Q summaries. The update is also not a standalone offline RL policy iteration step; it is a **narrow teacher-construction rule** that writes into Hydra’s already-existing policy loss path and `delta_q` head, exactly where reconciliation says the first tranche should focus. ([GitHub][1])

  ### 6. what is genuinely novel synthesis

  **Category: B**

  The irreducible synthesis is:

  1. take Hydra’s current AFBS shell exactly as it exists now,
  2. interpret root child visits as a **search-support signal**,
  3. convert root Q and visits into **lower-confidence action improvements**,
  4. project those improvements through the **current apprentice policy** rather than replacing it with raw search softmax,
  5. supervise the existing `delta_q` head with that same conservative correction.

  That combination is not just “use ExIt” and not just “do SPIBB.” It is a Hydra-specific improvement operator for a partial search teacher.

  ### 7. why it transfers to Hydra specifically

  Hydra already wants ExIt + pondering + search-as-feature, already has a `delta_q` head, and reconciliation already says the narrow rollout should prefer **ExIt target + `delta_q` + `safety_residual` activation first** without adding heads or broadening AFBS. This candidate does exactly that: it generates a **credible** `exit_target`-style policy teacher and a matching `delta_q_target` from the current AFBS shell, instead of waiting for full public-belief search to be alive end to end. ([GitHub][3])

  ### 8. exact mathematical formulation

  Let:

  * (s) be a Hydra decision state.
  * (a \in {1,\dots,46}) be a Hydra action.
  * (\pi_\theta(a\mid s)) be the masked current policy.
  * AFBS root statistics give:

    * root value (\hat Q_0(s)),
    * child value (\hat Q_a(s)) for each expanded legal child,
    * child visit count (n_a(s)),
    * total root visits (N(s)=\sum_a n_a(s)).

  Define a count-based uncertainty radius:
  [
  u_a(s)=c\sqrt{\frac{\log(1+N(s))}{1+n_a(s)}}
  ]

  Define the **trusted set**
  [
  T(s)={a : n_a(s)\ge n_{\text{trust}}}.
  ]

  Define the **lower-confidence improvement**:
  [
  A^-_a(s)=
  \begin{cases}
  \hat Q_a(s)-\hat Q_0(s)-\beta u_a(s), & a\in T(s)\ \text{and $a$ legal}\
  0, & \text{otherwise}
  \end{cases}
  ]

  This says: only consider search corrections on actions that AFBS actually supported; even there, subtract a confidence penalty based on child support.

  Now project this back onto the baseline policy:
  [
  \pi^\dagger(a\mid s)=
  \frac{\pi_\theta(a\mid s)\exp\left(\eta,[A^-*a(s)]*+\right)}
  {\sum_b \pi_\theta(b\mid s)\exp\left(\eta,[A^-*b(s)]*+\right)}
  ]

  This is the solution of the KL-regularized improvement problem
  [
  \max_{\pi}\ \sum_a \pi(a\mid s)[A^-*a(s)]*+ - \frac{1}{\eta}\mathrm{KL}!\left(\pi(\cdot\mid s),|,\pi_\theta(\cdot\mid s)\right).
  ]

  State-level activation gate:
  [
  g(s)=\mathbf 1!\left[\max_a A^-*a(s)>\delta_A\right]\cdot \mathbf 1!\left[N(s)\ge N*{\min}\right].
  ]

  Training loss:
  [
  L = L_{\text{base}}

  * \lambda_{\pi}, g(s),\mathrm{KL}!\left(\pi^\dagger(\cdot\mid s),|,\pi_\theta(\cdot\mid s)\right)
  * \lambda_{\Delta}, g(s),\frac{1}{|A_{\text{legal}}(s)|}\sum_a m_a(s),\rho!\left(\Delta q_\theta(s,a)-A^-_a(s)\right),
    ]
    where:

  - (m_a(s)\in{0,1}) is the legal-mask indicator,
  - (\rho) is Huber loss,
  - (\Delta q_\theta(s,a)) is Hydra’s existing `delta_q` head.

  Optional phase-2 extension, **not required for the minimum prototype**:
  [
  A^-_a(s)\leftarrow A^-_a(s)-\lambda_r [U^{\text{risk}}*a(s)-\kappa]*+
  ]
  if a credible upper confidence bound on deal-in risk is later exposed.

  I numerically sanity-checked the projection behavior on a toy example: increasing (\beta) suppresses a superficially best but high-uncertainty action and returns the projected teacher toward the baseline once no action retains positive lower-confidence advantage. That is the intended mechanism.

  ### 9. tensor shapes and affected network interfaces

  No new main model heads are required.

  Current Hydra shapes already include:

  * input observation: ([B,192,34]),
  * `policy_logits`: ([B,46]),
  * `delta_q`: ([B,46]),
  * `safety_residual`: ([B,46]),
  * `belief_fields`: ([B,16,34]),
  * `mixture_weight_logits`: ([B,4]). ([GitHub][10])

  Minimal added **target** tensors:

  * `conservative_exit_target`: ([B,46]),
  * `conservative_exit_mask`: ([B,46]) or reuse legal mask,
  * `delta_q_target`: ([B,46]),
  * `search_weight`: ([B,1]),
  * optional `afbs_visits`: ([B,46]) for debugging only.

  No inference-side interface change is required for the minimum prototype.

  ### 10. exact algorithm pseudocode

  ```text
  Algorithm: Visit-Bootstrapped Conservative ExIt

  Input:
    state s
    current masked policy pi_theta(.|s)
    AFBS root with root value q_root, child q[a], child visits n[a]
    legal mask m[a]
    hyperparams: c, beta, eta, n_trust, N_min, delta_A

  1. N <- sum_a n[a]
  2. For each action a in 1..46:
         if m[a] == 0:
             Aminus[a] <- 0
             continue
         if n[a] < n_trust:
             Aminus[a] <- 0
             continue
         u[a] <- c * sqrt(log(1 + N) / (1 + n[a]))
         Aminus[a] <- q[a] - q_root - beta * u[a]

  3. If N < N_min or max_a Aminus[a] <= delta_A:
         mark search target absent for this state
         stop

  4. For each legal action a:
         bonus[a] <- max(Aminus[a], 0)

  5. Construct projected teacher:
         t[a] <- pi_theta[a] * exp(eta * bonus[a]) for legal a
         t[a] <- 0 for illegal a
         pi_dagger <- t / sum_b t[b]

  6. Emit targets:
         conservative_exit_target <- pi_dagger
         delta_q_target <- Aminus
         search_weight <- 1

  Training step:
    loss <- base_losses(batch)
    if conservative_exit_target present:
        loss += lambda_pi * KL(conservative_exit_target || masked_policy)
        loss += lambda_delta * Huber(delta_q_pred, delta_q_target, legal_mask)
  ```

  ### 11. exact Hydra surfaces it would touch

  * `hydra-core/src/afbs.rs`
    expose a root snapshot API returning `root_q`, `child_q[action]`, `child_visits[action]`, legal action alignment, and stable serialization for label generation.

  * `hydra-train/src/data/sample.rs`
    add optional carriers for `conservative_exit_target`, `delta_q_target`, and `search_weight`.

  * `hydra-train/src/training/losses.rs`
    add a conservative-ExIt KL loss and wire it to the existing `delta_q` loss path.

  * `hydra-train/src/bc.rs` and/or `hydra-train/src/rl.rs`
    consume the new targets when present.

  * Optional later touch: `hydra-core/src/bridge.rs`
    only if you want to log AFBS support diagnostics into the tensor for later ablations.

  This is exactly in-family with reconciliation’s file-by-file checklist: `sample.rs`, `mjai_loader.rs`, `rl.rs`/`bc.rs`, then only minimal `bridge.rs` / `afbs.rs` plumbing. ([GitHub][1])

  ### 12. prototype path

  1. Instrument AFBS root export in `afbs.rs`.
  2. Build a **hard-state bank** of about 50k states from existing replay/self-play, stratified by the same doctrine Hydra already cares about: low top-2 gap, dangerous defense states, and similar “specialist search” regimes.
  3. For each state, run the current shallow AFBS teacher once and compute (A^-_a) and (\pi^\dagger).
  4. Fine-tune a frozen-or-mostly-frozen apprentice with only:

     * policy KL to `conservative_exit_target`,
     * `delta_q` Huber to `A^-`.
  5. Compare against:

     * no ExIt,
     * raw `Softmax(Q/\tau)` ExIt,
     * simple visit-thresholded raw ExIt.

  Use Hydra’s deterministic evaluation/seed-bank discipline for repeatable comparisons. ([GitHub][11])

  ### 13. benchmark plan

  **Offline benchmark first**

  Use a fixed hard-state bank and a stronger adjudicator than the training teacher:

  * preferred adjudicator: deeper AFBS on the same states,
  * fallback adjudicator: fixed-(N) seeded continuation rollouts if deeper AFBS is not discriminative enough.

  Metrics:

  * **accepted-state precision**: among states where the method activates, how often does the teacher-preferred action beat the baseline action under the adjudicator?
  * **negative-update rate**: how often does the induced update push probability toward an adjudicator-worse action?
  * **support-efficiency curve**: precision as a function of accepted-state coverage.
  * **`delta_q` calibration**: correlation and sign accuracy of predicted `delta_q` vs adjudicator deltas.

  **Online benchmark second**

  Equalize search-label budget and compare short self-play matches among:

  1. baseline,
  2. raw ExIt,
  3. visit-threshold raw ExIt,
  4. conservative ExIt.

  Use the fixed evaluation seed-bank and report both mean score utility and rank-sensitive metrics. ([GitHub][11])

  ### 14. what success would look like

  Success is **not** “slightly better loss curves.” Success is:

  * accepted search-labeled states are materially cleaner than raw ExIt,
  * negative updates drop noticeably,
  * the apprentice improves at equal or lower search-label volume,
  * and the gain survives comparison against the simpler “visit-threshold + raw search softmax” baseline.

  The qualitative success pattern I want is: *Hydra trusts search less often, but the states it does trust are much more worth trusting.*

  ### 15. what would kill the idea quickly

  Kill it fast if either of these happens:

  * A simple baseline of **raw ExIt plus a visit threshold** matches it within noise.
  * The adjudicator shows that AFBS child visits are a poor proxy for teacher reliability, so the lower-confidence correction mostly shrinks useful updates instead of filtering bad ones.

  ### 16. red-team failure analysis

  **How does this break in a 4-player general-sum game?**
  The SPIBB/CPI flavor does **not** inherit its formal guarantee here. AFBS values are model/search estimates in a multiplayer general-sum partially observed setting, not exact MDP action-values. So the method is conservative by mechanism, not by theorem.

  **Does this violate partial observability?**
  No. It only uses search outputs that Hydra itself generated from public state plus whatever current search context Hydra already uses. It does **not** assume hidden hands or fantasy observability.

  **Does it need labels Hydra does not have?**
  No for the minimum prototype. It only needs root Q and root child visits from AFBS plus the existing policy and `delta_q` surfaces. The optional risk-penalty extension does require more than the minimum prototype.

  **Is it secretly weaker than a simpler move already on the mainline?**
  Possibly. That is why the mandatory ablation is against **visit-thresholded raw ExIt**, not only against no ExIt.

  **Does it collapse into an incremental tuning trick once written out?**
  It might. If the whole gain can be reproduced by “just require (n_a\ge n_{\text{trust}}), then use raw search softmax,” I would demote it from separator candidate to tuning trick.

  ### 17. why this is more likely to matter than the strongest simpler mainline alternative

  The strongest simpler mainline alternative is exactly what `HYDRA_FINAL.md` points toward: hard-state-gated AFBS with raw ExIt targets from `Softmax(Q/\tau)`. The problem is that this changes only **where** search is used, not **how** search corrections are trusted. In Hydra’s current posture, teacher trustworthiness is the bottleneck. This candidate changes the improvement operator itself: unsupported or weakly supported search children cannot yank the apprentice around, while credibly improved children still can. That is a more structural lever than “search only on hard states.” ([GitHub][3])

  ### 18. closest known baseline and why this does not reduce to it

  Closest known baseline: **SPIBB-style conservative improvement around a baseline policy**, combined with **ExIt**. ([Proceedings of Machine Learning Research][6])

  Exact overlap:

  * conservative update around a baseline,
  * support-aware teacher trust,
  * planner-to-policy distillation.

  Irreducible difference:

  * support is **search visitation support**, not replay dataset action counts;
  * the operator lives inside a **46-action Riichi apprentice** with legal masks and a pre-existing `delta_q` head;
  * the teacher is not an offline RL critic but AFBS root statistics;
  * the goal is not a theorem of safe improvement, but a **search-target hygiene mechanism** for Hydra’s first live ExIt tranche.

  So this is **B**, not C.

  ### 19. dependency closure table: required signal / label / hook / teacher / runtime state | already exists / cheap to expose / missing | evidence or blocker

  | Required item                                | Status                          | Evidence / blocker                                       |
  | -------------------------------------------- | ------------------------------- | -------------------------------------------------------- |
  | `policy_logits` ([B,46])                     | already exists                  | main model head surface                                  |
  | `delta_q` ([B,46])                           | already exists                  | main model head surface                                  |
  | legal action mask ([B,46])                   | already exists                  | runtime + loader path                                    |
  | AFBS root child Q summaries                  | cheap to expose                 | AFBS shell exists; needs root snapshot export            |
  | AFBS root child visit counts                 | cheap to expose                 | AFBS shell exists; needs root snapshot export            |
  | `exit_target` production path                | missing in mainline             | reconciliation explicitly says it is absent              |
  | `delta_q_target` production path             | missing in mainline             | reconciliation explicitly says it is absent              |
  | `safety_residual_target`                     | already exists                  | replay-safe target builder already in loader             |
  | repeated posterior uncertainty beyond visits | missing for v1, optional later  | blocked by AFBS not yet being full public-belief runtime |
  | robust opponent archetype soft-min           | reserve / missing for prototype | reserve-shelf item, not first-tranche dependency         |

  Rows 1–3 are grounded in the current model/runtime surfaces; rows 4–7 in reconciliation’s repo-state checklist; row 8 in the current loader; rows 9–10 in reconciliation’s active-vs-reserve split. ([GitHub][2])

  ### 20. minimum falsifiable prototype

  A **two-weekend prototype**, not a quarter:

  * export AFBS root `q_root`, `child_q`, `child_visits`,
  * build 50k hard states,
  * generate `conservative_exit_target` and `delta_q_target`,
  * train only policy KL + `delta_q` Huber on top of a frozen baseline,
  * adjudicate on a held-out hard-state bank with deeper search.

  If that prototype cannot beat **visit-thresholded raw ExIt** on accepted-state precision and negative-update rate, reject the idea.

  ---

  ## Candidate 2

  ### 1. name

  **Value-of-computation routing for specialist compute and pondering**

  ### 2. problem solved

  Hydra’s doctrine is already selective-compute doctrine: AFBS should be specialist, hard-state gated; endgame exactification is conditional; pondering should use idle time; `HYDRA_FINAL.md` even names specific triggers like small top-2 policy gap, high-risk defense, low particle ESS, and wall-small endgame. The problem is that these are currently **heuristic gates**, not a learned compute-allocation policy. Hydra needs a router that decides **when extra compute is actually worth its latency** and, later, **which queued future states deserve pondering budget most**. ([GitHub][1])

  ### 3. outside ingredients and exact sources

  The outside ingredients are:

  * **Metareasoning for MCTS**: computation itself has value and should be selected by expected decision improvement, not applied uniformly. ([EECS at UC Berkeley][12])
  * **OCBA-style tree budget allocation**: limited search budget should be allocated to maximize probability of correct root selection, not just cumulative sampling regret. ([arXiv][13])
  * **Planning-learning trade-off**: performance peaks at an intermediate planning budget; neither “always plan” nor “never plan” is optimal. ([arXiv][14])

  ### 4. what is borrowed unchanged

  Borrowed unchanged:

  * Value-of-computation objective: extra compute is chosen only if expected utility gain exceeds its cost.
  * Budget-allocation framing: compute is a scarce resource to allocate under a constraint.
  * “Neither too fast nor too slow”: optimal policy is typically interior, not at the extremes.

  ### 5. what is adapted for Hydra

  The Hydra adaptation is that the “actions” of the metareasoner are not tree-expansion actions. They are **Hydra compute modes**:

  [
  \mathcal M = {\text{fast},\ \text{HandEV+},\ \text{shallow AFBS},\ \text{deep AFBS/ponder},\ \text{endgame exactifier}}.
  ]

  And the state signal is not generic MCTS statistics alone; it is Hydra’s actual public/runtime diagnostics: policy gap, danger structure, opp-tenpai signals, wall depth, score pressure, CT-SMC ESS, mixture entropy, Hand-EV disagreement, and whether search/belief context is present. That is very specific to Hydra’s current architecture and doctrine. ([GitHub][2])

  ### 6. what is genuinely novel synthesis

  **Category: B**

  The synthesis is a **single learned cost-sensitive compute policy** that serves both:

  * **on-turn routing**: do we stay fast-path or invoke a specialist module?
  * **ponder routing**: among queued predicted future states, which ones should consume idle-time budget and with which mode?

  That is not the same as per-node MCTS metareasoning and not the same as hand-coded top-2-gap gating. It is a Hydra-specific compute allocator over heterogeneous modules and future-state jobs.

  ### 7. why it transfers to Hydra specifically

  Hydra is already architected around a fast path, a slow search path, idle-time pondering, hard-position-only deepening, and latency-aware endgame triggers. `HYDRA_FINAL.md` gives the deployment budgets and the current intended hard-state heuristics; `bridge.rs` already has optional search/belief context fields; reconciliation says a fast-path vs ponder-cache split already exists. This candidate sharpens a doctrine Hydra already wants instead of trying to replace it. ([GitHub][3])

  ### 8. exact mathematical formulation

  For a state (s), let (m=0) denote the fast path, and (m\in\mathcal M\setminus{0}) denote extra compute modes.

  Let:

  * (a_m(s)) be the action chosen after running mode (m),
  * (u_m(s)) be an offline-estimated utility of choosing (a_m(s)),
  * (c_m(s)) be measured latency cost in milliseconds,
  * (\lambda > 0) be the utility-per-ms tradeoff coefficient.

  Define the net value of computation:
  [
  \Delta_m(s)=u_m(s)-u_0(s)-\lambda,c_m(s).
  ]

  The router takes diagnostics (x(s)\in\mathbb R^d) and predicts
  [
  r_\phi(x(s)) \in \mathbb R^{|\mathcal M|},
  \qquad
  r_\phi(x(s))_m \approx \Delta_m(s).
  ]

  Training loss, regression form:
  [
  L_{\text{router}}
  =================

  \sum_{m\in\mathcal M}
  w_m(s),\rho!\left(r_\phi(x(s))_m-\Delta_m(s)\right),
  ]
  with optional weights
  [
  w_m(s)=\frac{1}{\hat\sigma_m^2(s)+\epsilon}
  ]
  if the utility estimate comes from multiple seeded continuations.

  On-turn decision rule:
  [
  m^*(s)=\arg\max_{m\in\mathcal M} r_\phi(x(s))*m,
  \qquad
  \text{use fast path if } r*\phi(x(s))_{m^*(s)}\le 0.
  ]

  Ponder scheduling over a queue of candidate jobs ((s_j,m)):
  [
  \max_{y_{jm}\in{0,1}}
  \sum_{j,m} y_{jm},r_\phi(x(s_j))*m
  ]
  subject to
  [
  \sum*{j,m} y_{jm}c_{jm}\le B,
  \qquad
  \sum_m y_{jm}\le 1 \ \forall j,
  ]
  where (B) is idle-time budget.

  For a small queue this can be solved exactly as a knapsack; for a larger queue a greedy ratio (r/c) is enough for the first prototype.

  Numerical sanity check: the break-even condition is simply
  [
  u_m-u_0 > \lambda c_m.
  ]
  In a toy sweep, with (\lambda=0.05) utility-units/ms, a 25ms shallow-search call must clear (1.25) utility units, and a 100ms deep call must clear (5.0). That makes the cost term large enough to prevent degenerate “always search” routing.

  ### 9. tensor shapes and affected network interfaces

  Minimal prototype: **no change to the main Hydra network**.

  Use a separate tiny router:

  * diagnostic input (x(s)): ([B,d]), with (d=16) in the minimum prototype,
  * router output:

    * binary gate prototype: ([B,2]),
    * full mode router: ([B,|\mathcal M|]), where (|\mathcal M|=5).

  A concrete (d=16) feature set:

  1. policy entropy,
  2. top-1 probability,
  3. top-2 gap,
  4. value,
  5. wall remaining,
  6. score gap to next place,
  7. score gap to previous place,
  8. riichi-threat count,
  9. max opponent tenpai probability,
  10. min legal danger,
  11. mean legal danger,
  12. max legal danger,
  13. best legal `safety_residual`,
  14. Hand-EV best-vs-policy gap,
  15. CT-SMC ESS ratio,
  16. mixture entropy.

  Where unavailable, use zeros plus a small presence mask, consistent with Hydra’s “feature present vs zeroed” doctrine. Existing Hydra surfaces already provide the relevant policy, danger, tenpai, `safety_residual`, and search-context families. ([GitHub][2])

  ### 10. exact algorithm pseudocode

  ```text
  Algorithm: Value-of-Computation Router

  Offline label generation:
  Input:
    state bank S
    modes M = {fast, handev, shallow_afbs, deep_afbs, endgame}
    utility evaluator U_eval
    cost coefficient lambda

  For each state s in S:
      x[s] <- extract_runtime_diagnostics(s)

      For each mode m in M:
          start timer
          a_m <- action chosen by mode m on s
          c_m <- elapsed ms
          u_m <- U_eval(s, a_m)   # deeper teacher or fixed-seed continuation value
          delta_m <- u_m - u_fast - lambda * c_m

      store (x[s], delta_[1..|M|])

  Train router:
      r_phi(x) -> predicted delta per mode
      minimize weighted Huber regression to stored deltas

  Runtime on-turn:
      x <- extract_runtime_diagnostics(current state)
      delta_hat <- r_phi(x)
      m_star <- argmax(delta_hat)
      if delta_hat[m_star] <= 0:
          use fast path
      else:
          invoke mode m_star

  Runtime pondering:
      build queue of predicted near-future states
      for each queued state and each available mode:
          score <- r_phi(x_j)[m] / cost_jm
      greedily pick jobs until idle-time budget exhausted
  ```

  ### 11. exact Hydra surfaces it would touch

  * `hydra-core/src/bridge.rs`
    expose the diagnostic scalars and context-presence flags needed by the router.

  * `hydra-core/src/hand_ev.rs`
    expose a cheap summary statistic like “best Hand-EV minus policy-action Hand-EV.”

  * `hydra-core/src/afbs.rs`
    expose mode-level latency and root summary hooks for shallow/deep AFBS.

  * `hydra-core/src/ct_smc.rs`
    expose ESS ratio and maybe posterior-quality diagnostics already implied by the current sampler.

  * `hydra-core/src/endgame.rs`
    register endgame mode latency and activation eligibility.

  * `hydra-train/src/inference.rs` or the current runtime selection path
    consume router decisions for fast-path vs slow-path vs ponder scheduling.

  * `[new] hydra-core/src/compute_router.rs`
    a tiny MLP or even logistic regression wrapper for the first prototype.

  Reconciliation already notes that a fast-path vs ponder-cache split exists, and the north-star docs already specify hard-position triggers and deployment budgets; this candidate only replaces those heuristics with a learned cost-sensitive router. ([GitHub][1])

  ### 12. prototype path

  Start narrower than the full scheduler:

  1. Two-mode prototype only: **fast** vs **shallow AFBS**.
  2. Collect a stratified state bank from replay/self-play.
  3. Label each state with:

     * fast action utility,
     * shallow-AFBS action utility,
     * measured latency.
  4. Train a 16-input logistic regressor or 1-hidden-layer MLP to predict (\Delta_{\text{shallow}}).
  5. Compare against:

     * always fast,
     * always shallow AFBS,
     * hand-coded gate: `top-2 gap < 10% OR high-risk defense OR low ESS`.

  If that works, extend to full (|\mathcal M|=5) and finally to ponder-queue scheduling.

  ### 13. benchmark plan

  **Offline first**

  Use a fixed state bank and equalize utility evaluation:

  * Evaluate each mode’s chosen action with the **same adjudicator**:

    * deeper AFBS, or
    * fixed seeded continuation rollouts.

  Compare utility-latency frontiers:

  1. fast only,
  2. heuristic gate,
  3. learned router.

  Metrics:

  * area under utility-vs-latency frontier,
  * mean utility at fixed mean latency,
  * mean utility at fixed 95th percentile latency,
  * false-positive rate: router invoked extra compute but (\Delta_m < 0).

  **Online second**

  Use the evaluation seed bank and equalize **mean** and **P95** per-turn latency across agents. Then compare:

  * score utility,
  * average placement,
  * 4th-place rate,
  * fraction of turns that invoked each compute mode.

  Hydra’s seeding/testing docs make this kind of controlled comparison a first-class requirement. ([GitHub][11])

  ### 14. what success would look like

  Success is a **strictly better utility-latency frontier** than the heuristic gate. At fixed compute budget, the router should send compute to fewer but more leverage-heavy states. In the extended version, it should also improve the **quality of ponder labels per idle second**, not just runtime play.

  ### 15. what would kill the idea quickly

  Kill it if either of these happens:

  * a 3-feature threshold rule or logistic regression on `top-2 gap`, `danger`, and `wall remaining` matches it;
  * or the learned router beats the heuristic only by spending more compute in disguise, not by allocating the same budget better.

  ### 16. red-team failure analysis

  **How does this break in a 4-player general-sum game?**
  Utility labels are policy- and opponent-mixture-dependent. A router trained on one opponent mix may overfit and misallocate compute against another.

  **Does this violate partial observability?**
  No. It only uses public/runtime diagnostics Hydra already has or can cheaply expose. It does not assume hidden hands.

  **Does it require targets Hydra does not actually have now?**
  The minimum prototype does require **mode-utility labels**, which Hydra does not currently log in the mainline. But those labels are self-generated and cheap compared with new hidden-state supervision.

  **Is it secretly weaker than a simpler selective-compute move?**
  Possibly. If the hand-coded gate from `HYDRA_FINAL.md` is already near-optimal, this candidate is just polish.

  **Does it collapse into an incremental tuning trick?**
  It does if it cannot beat strong threshold baselines at equal budget. That is why the first benchmark must be a frontier comparison, not a raw Elo/self-play comparison.

  ### 17. why this is more likely to matter than the strongest simpler mainline alternative

  The strongest simpler alternative is exactly the heuristic already implicit in `HYDRA_FINAL.md`: more compute when top-2 gap is small, defense is risky, ESS is low, and wall is short. That is a good heuristic, but it is still a **static rule over a few marginals**. Hydra’s actual compute problem is contextual and combinatorial: different specialist modules pay off in different score-pressure regimes, wall phases, and opponent-threat structures, and pondering introduces a queue-allocation problem on top. A cost-sensitive learned router is the smallest mechanism that can solve that broader problem. ([GitHub][3])

  ### 18. closest known baseline and why this does not reduce to it

  Closest known baseline: **metareasoning / value-of-computation for MCTS** plus **OCBA-style sample allocation**. ([EECS at UC Berkeley][12])

  Exact overlap:

  * treat compute as an action,
  * optimize expected decision improvement under a budget,
  * allocate scarce budget where it matters most.

  Irreducible difference:

  * the actions here are **heterogeneous Hydra modules**, not more samples at one node;
  * the same learned router spans **on-turn routing and ponder scheduling**;
  * the signal space is Hydra’s real **policy/danger/ESS/Hand-EV/score-pressure** surface, not only tree-bandit statistics.

  So this is **B**, not C.

  ### 19. dependency closure table: required signal / label / hook / teacher / runtime state | already exists / cheap to expose / missing | evidence or blocker

  | Required item                                     | Status                       | Evidence / blocker                                |
  | ------------------------------------------------- | ---------------------------- | ------------------------------------------------- |
  | policy entropy / top-2 gap / value                | cheap to expose              | model outputs already exist                       |
  | danger / opp-tenpai / `safety_residual` summaries | already exists               | current head surface exists                       |
  | wall remaining / score pressure / riichi flags    | already exists               | runtime state already tracks them                 |
  | CT-SMC ESS / mixture entropy                      | cheap to expose              | search context already has CT-SMC / mixture hooks |
  | Hand-EV disagreement summary                      | cheap to expose              | Hand-EV already exists; needs summary hook        |
  | fast-path vs ponder-cache integration point       | already exists               | reconciliation says split exists                  |
  | per-mode latency logs                             | cheap to expose              | instrumentation needed, not conceptual blocker    |
  | per-mode utility labels                           | missing in mainline          | must be generated offline from adjudicator        |
  | tiny external router model                        | missing but trivial          | new helper file only                              |
  | full ponder knapsack scheduler                    | cheap to expose after router | second-stage extension, not prototype blocker     |

  Rows 1–5 are grounded in the current model, runtime, and bridge/search-context surfaces; row 6 in reconciliation’s runtime note; rows 7–10 are implementation tasks, not architectural blockers. ([GitHub][2])

  ### 20. minimum falsifiable prototype

  A single binary gate:

  * modes: fast vs shallow AFBS,
  * features: 16 diagnostics,
  * labels: (\Delta_{\text{shallow}} = u_{\text{shallow}} - u_{\text{fast}} - \lambda c),
  * model: logistic regression or tiny MLP,
  * benchmark: utility-latency frontier against the existing heuristic gate.

  If the binary router cannot beat the heuristic gate at equal budget, reject the broader compute-routing story.

  ---

  ## The single best candidate to try first

  **Candidate 1: Visit-bootstrapped conservative Expert Iteration.**

  It attacks Hydra’s most immediate bottleneck: not “missing more search,” but “extracting trustworthy improvement targets from the search shell Hydra already has,” which is exactly where reconciliation says the next tranche should live. ([GitHub][1])

  ## The single best cheap benchmark to run first

  A **hard-state offline adjudication benchmark** comparing:

  1. no ExIt,
  2. raw `Softmax(Q/\tau)` ExIt,
  3. visit-threshold raw ExIt,
  4. visit-bootstrapped conservative ExIt,

  with a deeper adjudicator on the same state bank, measuring accepted-state precision and negative-update rate. That benchmark is cheaper and more diagnostic than jumping straight to self-play.

  ## The single biggest hidden implementation risk

  **AFBS root Q semantics may be too weak or too unstable.**
  If `q_root` and `child_q` are not well enough aligned with actual action quality, then candidate 1’s conservative correction is calibrating noise, and candidate 2’s utility labels will also inherit that weakness. This is the deepest shared risk.

  ## The 2-5 most tempting rejected directions and exactly why they were rejected

  * **Immediate robust exploitation / safe subgame refinement from poker-style work**: tempting because Hydra has `robust_opponent` math and the outside literature has adaptation-safe exploitation results, but those results are centered on **two-player zero-sum** imperfect-information games, while reconciliation keeps robust-opponent search backups on the reserve shelf until supervision and local evaluators are healthier. ([Proceedings of Machine Learning Research][15])

  * **Full public-belief search as immediate mainline**: explicitly rejected by reconciliation as too expensive and too early relative to target closure. It is a later research branch, not the next coding tranche. ([GitHub][1])

  * **Raw Sinkhorn / tile-allocation belief supervision or more belief heads**: tempting because Mixture-SIB and belief surfaces are real, but reconciliation explicitly says no new heads now, and that if belief supervision is activated it should target projected/public-teacher objects or gauge-fixed marginals rather than raw latent allocations. `OPPONENT_MODELING.md` also marks heavier belief/call-intent/RSA-style additions as reserve. ([GitHub][1])

  * **Reviving DRDA/ACH as the immediate coding bet**: too early. Reconciliation keeps DRDA/ACH as north-star direction but says the first coding tranche should not depend on resolving optimizer-level debates first. ([GitHub][1])

  * **Call-intent / FiLM / RSA deception expansion**: strategically interesting, but explicitly reserve-shelf, head-expanding, and not minimum-falsifiable under current Hydra constraints. ([GitHub][16])

  [1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md"
  [2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/model.rs "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/model.rs"
  [3]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md"
  [4]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs"
  [5]: https://proceedings.neurips.cc/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html "https://proceedings.neurips.cc/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html"
  [6]: https://proceedings.mlr.press/v97/laroche19a.html "https://proceedings.mlr.press/v97/laroche19a.html"
  [7]: https://arxiv.org/abs/1709.10089 "https://arxiv.org/abs/1709.10089"
  [8]: https://arxiv.org/abs/2103.12021 "https://arxiv.org/abs/2103.12021"
  [9]: https://arxiv.org/abs/2411.00405 "https://arxiv.org/abs/2411.00405"
  [10]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md"
  [11]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md"
  [12]: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2011/EECS-2011-119.html "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2011/EECS-2011-119.html"
  [13]: https://arxiv.org/abs/2009.12407 "https://arxiv.org/abs/2009.12407"
  [14]: https://arxiv.org/abs/2005.07404 "https://arxiv.org/abs/2005.07404"
  [15]: https://proceedings.mlr.press/v235/ge24b.html "https://proceedings.mlr.press/v235/ge24b.html"
  [16]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md"
  ]]>
  </answer_text>
  </answer_section>
</combined_run_record>
