<combined_run_record run_id="008_invent_new_techniques" variant_id="agent_8_diagnostic_note" schema_version="1">
  <metadata>
    <notes>Later diagnostic note about prompt formatting; paired here for context but not a substantive run answer.</notes>
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
  <answer_text status="preserved_diagnostic" source_path="agent_8.md">
  <![CDATA[# Why PROMPT_7 and PROMPT_8 started producing bold labels with short paragraphs

  Yes — we pretty clearly told it to do something in that direction.

  ## Short answer

  The formatting is mostly coming from the prompt design, not from random model drift.

  Both `PROMPT_7_FRESH_CONTEXT_CROSS_FIELD_BREAKTHROUGH.md` and `PROMPT_8_FRESH_CONTEXT_INVENT_NEW_TECHNIQUES.md` explicitly push the model toward:

  - exact section ordering
  - compact output
  - information-dense writing
  - prototype-oriented answers instead of long narrative prose

  That combo strongly nudges GPT-5.4 Pro into a structured markdown brief style, which often shows up as:

  - a heading per major technique
  - bold field labels like `**Problem solved.**`
  - one short paragraph per field

  So yeah: we did not literally say "use bold labels," but we absolutely created the conditions that make the model choose that format.

  ## The strongest evidence from the prompt files

  From `PROMPT_7_FRESH_CONTEXT_CROSS_FIELD_BREAKTHROUGH.md`:

  - `Return exactly the requested sections, in the requested order.`
  - `Keep the answer compact, evidence-backed, and prototype-oriented.`
  - `Prefer concise, information-dense writing.`
  - `Return only 1-3 serious candidates.`
  - `For each candidate give:` followed by a field list

  From `PROMPT_8_FRESH_CONTEXT_INVENT_NEW_TECHNIQUES.md`:

  - `Return exactly the requested sections, in the requested order.`
  - `Keep the final answer compact, evidence-backed, and prototype-aware.`
  - `Prefer concise, information-dense writing.`
  - `Return only 1-3 serious techniques.`
  - `For each one, provide:` followed by a field list

  That is basically a template for "structured research brief" output instead of "verbose essay" output.

  ## Why the model chose bold labels specifically

  This part is probably a mix of prompt pressure plus GPT-5.4 default behavior.

  Our local prompt notes in `prompting.md` say:

  - `GPT-5.4 often defaults to more structured formatting and may overuse bullet lists.`

  And the Hydra prompt style guide says:

  - `explicit output contracts improve reliability`
  - `concise, information-dense output works better than vague verbosity`

  So once we told the model:

  1. be compact
  2. be exact about section order
  3. cover a fixed field list for each candidate
  4. avoid fluff

  the easiest stable markdown shape became:

  - section heading
  - bold field label
  - short explanatory paragraph

  That is a normal model response to this kind of contract-heavy prompt.

  ## Why it feels less verbose than older runs

  Because we deliberately optimized these later prompts for density and control.

  Compared with earlier prompt generations, these fresh-context prompts added stronger pressure around:

  - compactness
  - exact deliverable shape
  - rejection of scene-setting
  - only keeping 1-3 serious ideas
  - answering like something a coding agent could prototype

  That naturally compresses the prose.

  In plain English: we traded some "essay voice" for tighter engineering-brief voice.

  ## Did we explicitly request bold formatting?

  No, not explicitly.

  I do not see a line in `PROMPT_7` or `PROMPT_8` that says to use bold labels.

  But we did explicitly request the conditions that usually produce it:

  - rigid section structure
  - concise output
  - dense information packing
  - markdown-friendly deliverables

  So the answer is:

  - **Did we directly command bold labels?** No.
  - **Did we strongly steer the model into that style?** Yes.

  ## If you want the old more verbose style back

  Then the prompts should stop over-constraining brevity.

  The biggest lines causing compression are the ones like:

  - `Keep the answer compact...`
  - `Prefer concise, information-dense writing.`
  - `Return exactly the requested sections...`

  If we want more expansive outputs next time, the prompt should instead say something more like:

  - explain each candidate in full prose, not label-style fragments
  - use multi-paragraph reasoning per candidate
  - include a longer comparative discussion before the final ranking
  - do not optimize for compactness if detail would help implementation

  ## Bottom line

  The weird part is not that GPT-5.4 got randomly shorter.

  The real reason is that our prompt stack for `PROMPT_7` and `PROMPT_8` was tuned to produce disciplined, compact, section-locked engineering briefs. GPT-5.4 then rendered that in a common markdown pattern: bold labels followed by short paragraphs.

  So yes — this output style is mostly downstream of our own instructions.
  ]]>
  </answer_text>
  </answer_section>
</combined_run_record>
