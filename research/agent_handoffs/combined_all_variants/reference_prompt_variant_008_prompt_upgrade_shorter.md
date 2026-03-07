<combined_run_record run_id="008_invent_new_techniques" variant_id="prompt_upgrade_variant_shorter" schema_version="1">
  <metadata>
    <notes>Historical prompt-only variant from the prompt-upgrade worktree. Main root prompt is newer; this is preserved for context before deleting the worktree.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved_from_worktree" source_path=".worktrees/prompt-upgrade/PROMPT_8_FRESH_CONTEXT_INVENT_NEW_TECHNIQUES.md" kind="historical_prompt_variant">
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

  If you cannot make an idea technically crisp, kill it.

  <output_contract>
  - Return exactly the requested sections, in the requested order.
  - Be as detailed and explicit as necessary; do not optimize for brevity.
  - Return a full technical treatment, not a compressed memo.
  - Return only 1-3 serious techniques.
  </output_contract>

  <verbosity_controls>
  - Prefer full technical exposition over compressed summary.
  - Use multi-paragraph explanations where needed.
  - Do not omit equations, derivations, tensor/interface details, pseudocode, assumptions, thresholds, edge cases, or implementation caveats when they matter.
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
  - Only present techniques that survive this adversarial self-review.
  </self_red_team_rules>

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
  3. Invent only the smallest number of serious techniques that look genuinely new for Hydra and technically workable.
  4. For each candidate, write down the objective/update rule, tensor interfaces, algorithm pseudocode, repo insertion points, and cheapest falsification path.
  5. Try hard to kill each idea before keeping it.

  ## Deliverables

  Return only 1-3 serious techniques.

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

  Then end with:
  - the single best technique to try first
  - the single best cheap benchmark to run first
  - the single biggest hidden implementation risk

  ## Hard constraints

  - no generic bigger-model or bigger-search proposals
  - no AGPL-derived code advice
  - no broad architecture resets
  - no pretty but untestable inventions
  - no keeping weak options alive just because they sound novel
  - no shallow acronym invention
  - no recommendation that cannot be written as an executable objective/update rule plus pseudocode inside Hydra constraints

  Success means your final answer feels like a serious long-think invention pass and leaves behind at least one genuinely new Hydra technique that is mathematically explicit, red-teamed, and prototypeable.
  ]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="context_note" source_path="assistant_generated_context_note">
  <![CDATA[This artifact preserves a prompt-only historical variant recovered from an extra worktree before cleanup. No paired answer body was stored alongside this specific prompt variant.
  ]]>
  </answer_text>
  </answer_section>
</combined_run_record>
