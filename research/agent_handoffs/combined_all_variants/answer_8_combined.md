<combined_run_record run_id="008_invent_new_techniques" variant_id="agent_8_diagnostic_note" schema_version="1">
<metadata>
<notes>Later diag note on prompt formatting; paired here for context, not substantive run answer.</notes>
<layout>single_markdown_file_prompt_then_answer</layout>
</metadata>

<prompt_section>
<prompt_text status="preserved" source_path="PROMPT_8_FRESH_CONTEXT_INVENT_NEW_TECHNIQUES.md">
<![CDATA[# Hydra fresh-context prompt — invent genuinely new techniques with math and red-team pressure

Primary source material lives in raw GitHub links below.

  ## Critical directive — how to read the core Hydra docs

Must avoid known bad behavior: fragmented keyword-peeking over large architecture docs.

Bad behavior for this task:
  - keyword search first
  - read isolated 20-100 line chunks around keywords
  - treat docs like logs or grep DB
  - invent new techniques before holistic Hydra understanding

For this task, behavior above = disqualifying.

Required reading workflow:
  1. Use browse/fetch tool on raw GitHub links for core docs below.
  2. Read core docs holistically, sequentially, before narrower search.
  3. Build high-level model of Hydra active path, reserve shelf, runtime structure, training surfaces, already-partially-implemented loops.
  4. Only then use narrower search for exact details and outside inspiration.

Do not use grep-style keyword hunting as primary reading strategy for core docs.

<holistic_ingestion_rules>
  - Read core docs as whole docs before narrowing.
  - Do not start with keyword search on core docs.
  - Do not rely on fragmented line-window retrieval for architecture understanding.
  - After holistic reading, targeted search for exact details OK.
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

You act as long-think breakthrough engineer for Hydra.

Job: discover genuinely new techniques for Hydra, not renamed known tricks or shiny repackaging. May combine ingredients from different papers/systems only when mechanism becomes mathematically explicit, fits Hydra architecture, survives adversarial self-review.

Target not mere novelty. Target = separator-level breakthrough: technique that could matter for Hydra like LuckyJ signature ACH/search-era breakthroughs mattered for LuckyJ. Do not mechanically imitate ACH. Search for breakthrough of comparable strategic importance for Hydra actual architecture and bottlenecks.

If idea not technically crisp, kill it.

<output_contract>
  - Return exactly requested sections, requested order.
  - Be as detailed and explicit as needed; do not optimize for brevity.
  - Return full technical treatment, not compressed memo.
  - Return only 1-3 serious techniques.
  - Short answer usually = failure mode for this prompt.
</output_contract>

<verbosity_controls>
  - Prefer full technical exposition over compressed summary.
  - Use multi-paragraph explanations where needed.
  - Do not omit equations, derivations, tensor/interface details, pseudocode, assumptions, thresholds, edge cases, impl caveats when they matter.
  - When unsure, include more math detail, derivation, mechanism detail.
</verbosity_controls>

<research_mode>
  - Work in 3 passes:
    1. Ingest: read Hydra docs holistically, reconstruct real current mainline, reserve shelf, missing closures.
    2. Retrieve: search broadly for ingredient families, neighboring mechanisms, counterexamples.
    3. Synthesize: keep only techniques both genuinely novel for Hydra and technically viable under Hydra constraints.
  - Stop only when more searching likely will not change final ranking.
</research_mode>

<tool_persistence_rules>
  - Prefer full-document browse/fetch for core docs over fragmented terminal-style chunk reading.
  - Search beyond already-surfaced papers when that could materially improve novelty or falsification.
  - Do not stop at first plausible invention.
</tool_persistence_rules>

<calculation_validation_rules>
  - If rec depends on quantitative reasoning, derive explicitly.
  - Use executable arithmetic or small scripts when needed to sanity-check formulas, tensor shapes, threshold logic, algorithm invariants.
  - Do not fake arithmetic that could have been checked.
</calculation_validation_rules>

<dependency_checks>
  - Before proposing impl, verify Hydra already has or could cheaply expose needed signals, labels, runtime hooks.
  - Before proposing new objective or target, check whether needed trajectories, teacher outputs, hidden-state labels exist or can be safely derived.
</dependency_checks>

<posture_reconstruction_rules>
  - Before proposing any technique, include short "Hydra posture reconstruction" section with 5-10 bullets.
  - Bullets must distinguish current mainline doctrine, reserve-shelf ideas, partially closed loops, and 2-3 non-goals or deprioritized paths.
  - Do not propose breakthrough candidates until posture reconstruction complete.
</posture_reconstruction_rules>

<citation_rules>
  - Cite only sources retrieved in this workflow or sources included in raw links above.
  - Never fabricate references.
  - Attach citations to exact claims they support.
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
  - Do not invent shallow, buzzword-heavy acronyms.
  - If you propose novel technique, must prove technical viability.
  - In thinking space, explicitly write out:
    - mathematical formulation (ex: exact loss fn or Bellman update)
    - tensor shapes in and out of network
    - exact pseudocode for algorithm
  - If technique cannot be mathematically defined under Hydra architecture constraints, discard.
  - Do not confuse renamed known trick with genuinely new mechanism.
  - Be explicit: what borrowed unchanged, what adapted, what newly proposed.
</novelty_viability_rules>

<self_red_team_rules>
  - Before finalizing rec, Red-Team own ideas.
  - For every proposed technique, spend compute time actively searching provided docs for reasons idea fails.
  - Ask explicitly:
    - How break in 4-player general-sum game?
    - Does this violate partial observability constraints?
    - Does this require labels, targets, or privileged signals Hydra does not have?
    - Is claimed novelty fake because method collapses back to known technique under Hydra constraints?
    - Does simpler existing Hydra path already dominate this?
  - - Does supposed breakthrough collapse into incremental tuning trick once written mathematically?
  - Only present techniques that survive adversarial self-review.
</self_red_team_rules>

<anti_survey_rules>
  - Do not return literature survey, field map, or long adjacent-idea list without convergence.
  - Every cited outside paper, repo, or mechanism must earn place by changing final candidate set or red-team analysis.
  - If paragraph does not help define, falsify, compare, or prototype surviving candidate, cut it.
</anti_survey_rules>

<novelty_honesty_rules>
  - For every surviving technique, include "closest known baseline" subsection.
  - State nearest known method/family, exact overlap, irreducible difference.
  - If method reduces to known technique under realistic Hydra constraints, downgrade or reject.
  - Label each surviving candidate as one of:
    - `A`: genuinely new mechanism
    - `B`: known mechanism with Hydra-specific adaptation that plausibly changes capability
    - `C`: renamed or lightly modified known trick
  - Reject all `C` candidates.
</novelty_honesty_rules>

<minimum_falsification_rules>
  - For every surviving technique, define minimum falsifiable prototype that tests claimed breakthrough mechanism in isolation.
  - If core claim cannot be tested without large coupled rollout or major stack build-out, reject as too diffuse.
  - First benchmark should distinguish idea from stronger tuning, more search, more data, or easier teacher signals.
</minimum_falsification_rules>

<completeness_contract>
  - Treat task incomplete until every surviving technique includes exact mechanism, math formulation, tensor shapes, pseudocode, repo insertion points, cheapest prototype path, benchmark plan, kill criteria.
  - Mark any underspecified item [blocked] rather than pretending ready.
</completeness_contract>

<verification_loop>
  - Before finalizing, verify you read core Hydra docs holistically before narrowing in.
  - Verify each surviving technique not merely renamed known trick.
  - Verify each surviving technique mathematically defined strongly enough that coding agent could begin prototyping it.
  - Verify novelty claim survives own red-team pass.
</verification_loop>

<dig_deeper_nudge>
  - Do not stop at first cool invention.
  - Prefer capability-changing mechanisms over cosmetic complexity.
  - Search especially hard around belief compression, decision-focused uncertainty, teacher trust gating, opponent-mixture optimization, selective compute, multiplayer credit assignment, privileged-to-public distillation.
</dig_deeper_nudge>

  ## What to do

  1. Reconstruct Hydra real current posture before inventing anything.
  2. Search for outside ingredients, neighboring ideas, counterexamples.
  3. Invent only smallest number of serious techniques that look genuinely new for Hydra, technically workable, plausibly capable of separator-level breakthrough rather than minor improvement.
  4. For each candidate, write objective/update rule, tensor interfaces, algorithm pseudocode, repo insertion points, cheapest falsification path.
  5. Compare each surviving candidate against strongest simpler mainline alternative and reject if mostly incremental.
  6. Try hard to kill each idea before keeping it.

  ## Deliverables

Return only 1-3 serious techniques.

If no candidate survives novelty, grounding, prototypeability filters, return `0 surviving candidates` and explain why.

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
  15. what would kill idea quickly
  16. red-team failure analysis
  17. why this more likely to matter than strongest simpler mainline alternative
  18. closest known baseline and why this does not reduce to it
  19. dependency closure table: required signal / label / hook / teacher / runtime state | already exists / cheap to expose / missing | evidence or blocker
  20. minimum falsifiable prototype

Then end with:
  - single best technique to try first
  - single best cheap benchmark to run first
  - single biggest hidden impl risk
  - 2-5 most tempting rejected directions and exactly why rejected

  ## Hard constraints

  - no generic bigger-model or bigger-search proposals
  - no AGPL-derived code advice
  - no broad architecture resets
  - no pretty but untestable inventions
  - no keeping weak options alive only because prompt implicitly demanded at least one answer
  - no shallow acronym invention
  - no rec that cannot be written as executable objective/update rule plus pseudocode inside Hydra constraints
  - no answer that mainly summarizes literature instead of converging to prototype-worthy separator candidate
  - no candidate that survives only because prompt implicitly demanded at least one answer

Success means final answer feels like serious long-think invention pass and leaves at least one genuinely new Hydra technique that is mathematically explicit, red-teamed, prototypeable, and plausibly capable of becoming LuckyJ-caliber separator for Hydra rather than another incremental trick.
]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="preserved_diagnostic" source_path="agent_8.md">
<![CDATA[# Why PROMPT_7 and PROMPT_8 started producing bold labels with short paragraphs

Yes — pretty clear: prompt told model to move that way.

  ## Short answer

Formatting mostly comes from prompt design, not random model drift.

Both `PROMPT_7_FRESH_CONTEXT_CROSS_FIELD_BREAKTHROUGH.md` and `PROMPT_8_FRESH_CONTEXT_INVENT_NEW_TECHNIQUES.md` explicitly push model toward:

  - exact section ordering
  - compact output
  - information-dense writing
  - prototype-oriented answers, not long narrative prose

Combo above strongly nudges GPT-5.4 Pro into structured markdown brief style, often showing up as:

  - heading per major technique
  - bold field labels like `**Problem solved.**`
  - one short paragraph per field

So: we did not literally say "use bold labels," but we absolutely created conditions that make model choose that format.

  ## The strongest evidence from the prompt files

From `PROMPT_7_FRESH_CONTEXT_CROSS_FIELD_BREAKTHROUGH.md`:

  - `Return exactly the requested sections, in the requested order.`
  - `Keep the answer compact, evidence-backed, and prototype-oriented.`
  - `Prefer concise, information-dense writing.`
  - `Return only 1-3 serious candidates.`
  - `For each candidate give:` followed by field list

From `PROMPT_8_FRESH_CONTEXT_INVENT_NEW_TECHNIQUES.md`:

  - `Return exactly the requested sections, in the requested order.`
  - `Keep the final answer compact, evidence-backed, and prototype-aware.`
  - `Prefer concise, information-dense writing.`
  - `Return only 1-3 serious techniques.`
  - `For each one, provide:` followed by field list

That templates "structured research brief" output, not "verbose essay" output.

  ## Why the model chose bold labels specifically

This part likely = prompt pressure + GPT-5.4 default behavior.

Local prompt notes in `prompting.md` say:

  - `GPT-5.4 often defaults to more structured formatting and may overuse bullet lists.`

Hydra prompt style guide says:

  - `explicit output contracts improve reliability`
  - `concise, information-dense output works better than vague verbosity`

So once we told model:

  1. be compact
  2. be exact about section order
  3. cover fixed field list for each candidate
  4. avoid fluff

easiest stable markdown shape became:

  - section heading
  - bold field label
  - short explanatory paragraph

Normal model response to this kind of contract-heavy prompt.

  ## Why it feels less verbose than older runs

Because later prompts deliberately optimized for density and control.

Compared with earlier prompt generations, these fresh-context prompts added stronger pressure around:

  - compactness
  - exact deliverable shape
  - rejection of scene-setting
  - keeping only 1-3 serious ideas
  - answering like something coding agent could prototype

That naturally compresses prose.

Plain English: we traded some "essay voice" for tighter engineering-brief voice.

  ## Did we explicitly request bold formatting?

No, not explicitly.

I do not see line in `PROMPT_7` or `PROMPT_8` saying use bold labels.

But we did explicitly request conditions that usually produce it:

  - rigid section structure
  - concise output
  - dense information packing
  - markdown-friendly deliverables

So answer:

  - **Did we directly command bold labels?** No.
  - **Did we strongly steer model into that style?** Yes.

  ## If you want the old more verbose style back

Then prompts should stop over-constraining brevity.

Biggest lines causing compression:

  - `Keep the answer compact...`
  - `Prefer concise, information-dense writing.`
  - `Return exactly the requested sections...`

If we want more expansive outputs next time, prompt should instead say something like:

  - explain each candidate in full prose, not label-style fragments
  - use multi-paragraph reasoning per candidate
  - include longer comparative discussion before final ranking
  - do not optimize for compactness if detail helps impl

  ## Bottom line

Weird part not that GPT-5.4 randomly got shorter.

Real reason: prompt stack for `PROMPT_7` and `PROMPT_8` tuned for disciplined, compact, section-locked engineering briefs. GPT-5.4 then rendered that in common markdown pattern: bold labels followed by short paragraphs.

So yes — output style mostly downstream of our own instructions.
]]>
</answer_text>
</answer_section>
</combined_run_record>