# Hydra follow-up for agent 6 — DEBC-AR code-ready specification

This is the **no-zip, browse-tool-first** version of the prompt.

Assume you have access to a browse/fetch tool that can read raw GitHub markdown pages directly. Do **not** assume zip attachments are available.

Your primary project-reading path is the raw GitHub links below.

## Critical directive — how to read the core Hydra docs

You must avoid a known bad behavior: fragmented keyword-peeking over large architecture docs.

Bad behavior for this task:
- searching for keywords first
- reading isolated 20-100 line chunks around those keywords
- treating the docs like logs or a grep database
- designing DEBC-AR from scattered snippets instead of whole-system understanding

For this task, that behavior is disqualifying.

Required reading workflow:
1. Use your **browse/fetch tool on the raw GitHub links** for the core docs listed below.
2. Read those core docs **holistically and sequentially** before doing narrower searching.
3. Build a high-level model of Hydra's current search/belief/runtime stack.
4. Only after that may you use narrower searching for exact APIs, structs, and insertion points.

Do **not** use grep-style keyword hunting as your primary reading strategy for these core docs.

Core docs that must be read holistically first:
- `research/design/HYDRA_RECONCILIATION.md`
- `research/design/HYDRA_FINAL.md`
- `docs/GAME_ENGINE.md`
- `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md`
- `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md`
- `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md`

Only after the core docs are ingested holistically may you narrow in on:
- `hydra-core/src/ct_smc.rs`
- `hydra-core/src/afbs.rs`
- `hydra-core/src/bridge.rs`
- `hydra-core/src/endgame.rs`
- `hydra-core/src/hand_ev.rs`
- outside papers on belief compression / scenario reduction / metareasoning

<holistic_ingestion_rules>
- Read the core docs as whole documents before narrowing.
- Do not start with keyword search on the core docs.
- Do not rely on fragmented line-window retrieval for architecture understanding.
- After holistic reading, you may use targeted search for exact details.
</holistic_ingestion_rules>

## Reading order — use browse/fetch in this exact sequence

Holistic core-doc pass:
1. `research/design/HYDRA_RECONCILIATION.md`
2. `research/design/HYDRA_FINAL.md`
3. `docs/GAME_ENGINE.md`
4. `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md`
5. `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md`
6. `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md`

Then targeted implementation grounding:
7. `hydra-core/src/ct_smc.rs`
8. `hydra-core/src/afbs.rs`
9. `hydra-core/src/bridge.rs`
10. `hydra-core/src/endgame.rs`
11. `hydra-core/src/hand_ev.rs`
12. `hydra-train/src/data/sample.rs`
13. `hydra-train/src/data/mjai_loader.rs`
14. `hydra-train/src/training/losses.rs`
15. `hydra-train/src/model.rs`

Only after that should you branch into outside papers and GitHub examples.

## Raw GitHub fallback links

Core Hydra docs:
- `README.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/README.md
- `research/design/HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `research/design/HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `research/design/HYDRA_ARCHIVE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_ARCHIVE.md
- `research/design/IMPLEMENTATION_ROADMAP.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/IMPLEMENTATION_ROADMAP.md
- `research/design/OPPONENT_MODELING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md
- `docs/GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md

Thin source slices:
- `hydra-core/src/afbs.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/afbs.rs
- `hydra-core/src/bridge.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs
- `hydra-core/src/ct_smc.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/ct_smc.rs
- `hydra-core/src/endgame.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/endgame.rs
- `hydra-core/src/hand_ev.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/hand_ev.rs
- `hydra-train/src/data/sample.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/sample.rs
- `hydra-train/src/data/mjai_loader.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/mjai_loader.rs
- `hydra-train/src/training/losses.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/training/losses.rs
- `hydra-train/src/model.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/model.rs

Prior answer anchors:
- `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md

Reference links already surfaced as relevant:
- ISMCTS / hidden-information search pathologies — https://www.aifactory.co.uk/newsletter/2013_01_reduce_burden.htm
- Problem-driven / decision-based scenario reduction anchor — https://arxiv.org/abs/2404.07810
- Metareasoning / utility-of-computation — https://www2.eecs.berkeley.edu/Pubs/TechRpts/2011/EECS-2011-119.pdf
- Limited contingency planning anchor — https://icaps03.icaps-conference.org/satellite_events/documents/WS/WS2/09/Dearden.pdf

You previously proposed DEBC-AR and BCPP for Hydra. This time do not give another invention memo.

You are GPT-5.4 Pro operating in a long-horizon research-and-engineering mode. Favor disciplined retrieval, code-grounded synthesis, and strong completion behavior over fast ideation.

Take **DEBC-AR only** and turn it into a code-ready engineering and experimentation specification for the real Hydra repo.

You are allowed and encouraged to search outside papers, official docs, and remote repos for stronger ingredients, better clustering choices, better refinement logic, better failure checks, and better benchmark design. But your final output must converge to an implementable Hydra plan.

<output_contract>
- Return exactly the requested sections, in the requested order.
- Keep the answer compact, high-signal, and implementation-oriented.
- Do not pad with broad background once the mechanism is clear.
</output_contract>

<verbosity_controls>
- Prefer concise, information-dense writing.
- Do not shorten the answer so much that symbols, structs, thresholds, or benchmarks become ambiguous.
</verbosity_controls>

<research_mode>
- Work in 3 passes:
  1. Ingest: read the core Hydra docs holistically first using the raw links or equivalent full-document browsing.
  2. Retrieve: identify the unresolved engineering questions, then search for strong ingredients, alternatives, and failure cases with 1-2 second-order leads where useful.
  3. Synthesize: converge on one final design and reject weaker branches.
- Stop only when more searching is unlikely to materially improve the prototype spec.
</research_mode>

<tool_persistence_rules>
- Prefer full-document browse/fetch for core docs over fragmented terminal-style chunk reading.
- Use additional retrieval whenever it materially improves mechanism quality, implementation detail, or falsification.
- Do not stop at the first paper or first plausible clustering scheme.
</tool_persistence_rules>

<dependency_checks>
- Before naming an API or signal, verify Hydra already has it or could cheaply derive it.
- Do not assume a convenient head, trace, or feature exists until you verify it.
</dependency_checks>

<completeness_contract>
- Treat the task as incomplete until all required definitions, structs, thresholds, and benchmark details are present or explicitly marked [blocked].
- If an important term remains undefined, the answer is not done.
</completeness_contract>

<empty_result_recovery>
- If the first retrieval pass around belief compression, scenario reduction, or clustering is weak, try adjacent fields and stronger technical sources before concluding there is no better ingredient.
</empty_result_recovery>

<citation_rules>
- Cite only sources actually retrieved in this workflow or from the raw links below.
- Never fabricate references.
- Attach citations to the exact mechanism or claim they support.
</citation_rules>

<grounding_rules>
- Base repo claims only on the raw links below or other evidence you actually retrieved.
- Label inference as inference.
- State conflicts explicitly when sources disagree.
</grounding_rules>

<verification_loop>
- Before finalizing, check:
  - did you actually read the core Hydra docs holistically before narrowing in?
  - is `Q_fast` actually defined concretely?
  - are the data structures and loop semantics explicit enough to prototype?
  - are failure modes and kill criteria sharp rather than decorative?
  - would a coding agent still have to guess major interfaces?
</verification_loop>

<dig_deeper_nudge>
- Do not stop at the first plausible compression trick.
- Look for second-order failure modes, tail-risk handling, and benchmark traps.
- If the answer still reads like a research memo instead of a build spec, keep going.
</dig_deeper_nudge>

## Minimum bar for completion

Your answer is incomplete unless it contains all of the following:

1. an exact definition of the cheap proxy `Q_fast(I, X_p, a)` using current Hydra signals or a clearly justified small extension
2. exact action-space assumptions and legal-action masking rules
3. exact data shapes / structs for:
   - particle signature
   - particle summary
   - cluster record
   - medoid / representative record
   - refinement queue item
4. an explicit algorithm loop for root search with:
   - signature generation
   - clustering
   - representative evaluation
   - split/refine triggers
   - visit allocation
   - fallback behavior
5. a precise catastrophic-defense tail rule
6. default hyperparameters plus safe starting ranges
7. file-level Hydra insertion points and likely function boundaries
8. the cheapest serious benchmark protocol and exact success / kill thresholds

## Strong guidance

- You may revise the old DEBC-AR design if a better variant emerges from searching.
- You may combine ideas from multiple papers if that yields a stronger Hydra-specific mechanism.
- You may reject parts of the original design if they are too fragile.
- You may use pseudocode, API sketches, and data-flow diagrams in plain markdown.

## What to ground in repo reality

Ground your design in the real files when relevant:
- `hydra-core/src/ct_smc.rs`
- `hydra-core/src/afbs.rs`
- `hydra-core/src/bridge.rs`
- `hydra-core/src/endgame.rs`
- `hydra-core/src/hand_ev.rs`

Do not redesign Hydra from scratch. Work with the current system and strengthen it.

## Required output format

1. what DEBC-AR is trying to buy Hydra that current AFBS + CT-SMC does not
2. revised final mechanism
3. exact state/action/proxy definitions
4. exact data structures and API sketch
5. search loop pseudocode
6. defaults and thresholds
7. failure modes and safeguards
8. benchmark design and kill criteria
9. final verdict: prototype now / reserve shelf / reject

## Hard constraints

- no generic invention memo
- no hand-wavy symbols without definitions
- no pretending a term is precise when the repo cannot support it yet
- no AGPL code borrowing

Success means a coding agent could begin prototyping from your answer with only small amounts of repo-local judgment.
