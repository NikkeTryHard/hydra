# Hydra long-think prompt pack 6 — invention, synthesis, and prototype pressure

This file contains stronger follow-up prompt variants for the expensive external model on the old `prompt-6` lane.

The goal here is not just better ideas. The goal is to force deeper search, stronger synthesis, and outputs that are detailed enough to turn into real Hydra experiments or code.

These variants are tuned for **GPT-5.4 Pro / long-think GPT-5.4-style models**:

- explicit output contracts
- completion and verification gates
- disciplined retrieval + synthesis phases
- stronger anti-shallowness nudges
- compact, auditable final deliverables

---

## Shared attachment package for all variants

Attach these files if your UI allows multiple attachments:

- `hydra_breakthrough_docs_pack.zip`
- `hydra_breakthrough_thin_source_pack.zip`
- `deep_agent_20_pdfs.zip`

What each attachment contains:

- `hydra_breakthrough_docs_pack.zip`
  - the core Hydra doctrine, design, archive, and handoff docs
  - prior answer archive
  - tracked prompt templates
- `hydra_breakthrough_thin_source_pack.zip`
  - the docs pack plus thin source slices from the real Rust codebase
  - enough code to make repo-grounded design claims and draft APIs
- `deep_agent_20_pdfs.zip`
  - local paper/reference PDFs already collected for Hydra

If the attachments are inaccessible or incomplete, use the raw GitHub links below with your browse tool.

### Raw GitHub fallback links

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

Prior answers and inspiration anchors:

- `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md

Reference links already surfaced as relevant to this lane:

- ISMCTS / hidden-information search pathologies — https://www.aifactory.co.uk/newsletter/2013_01_reduce_burden.htm
- Problem-driven / decision-based scenario reduction and belief compression anchor used by the old answer — https://arxiv.org/abs/2404.07810
- Metareasoning / utility-of-computation anchor — https://www2.eecs.berkeley.edu/Pubs/TechRpts/2011/EECS-2011-119.pdf
- Limited contingency planning anchor — https://icaps03.icaps-conference.org/satellite_events/documents/WS/WS2/09/Dearden.pdf

---

## Variant A — DEBC-AR deep follow-up, code-ready and benchmark-ready

Use this if you want the single highest-value follow-up on the old prompt-6 result.

```md
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
  1. Plan: identify the unresolved engineering questions blocking DEBC-AR from becoming prototypeable.
  2. Retrieve: search for strong ingredients, alternatives, and failure cases; follow 1-2 second-order leads where useful.
  3. Synthesize: converge on one final design and reject weaker branches.
- Stop only when more searching is unlikely to materially improve the prototype spec.
</research_mode>

<tool_persistence_rules>
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
- Cite only sources actually retrieved or supplied in the attachments/raw links.
- Never fabricate references.
- Attach citations to the exact mechanism or claim they support.
</citation_rules>

<grounding_rules>
- Base repo claims only on the provided Hydra docs/code.
- Label inference as inference.
- State conflicts explicitly when sources disagree.
</grounding_rules>

<verification_loop>
- Before finalizing, check:
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

## Working expectations

- Think like a hybrid research scientist + senior staff engineer.
- Explore widely first, then converge hard.
- Verify important claims against multiple sources when possible.
- Try to break your own design before finalizing.
- Do not stop when you have a cool idea. Stop only when the idea has enough engineering shape to prototype.

## Problem framing

Hydra already has:

- CT-SMC and belief machinery
- AFBS and search reuse/caching
- predictive pondering
- Hand-EV planes
- selective-compute doctrine

The old DEBC-AR answer is interesting because it tries to compress hidden worlds by **decision consequence** rather than raw tile similarity. That is promising, but still too vague for code.

Your job is to fix that.

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
```

---

## Variant B — BCPP deep follow-up, with idle-time and reuse economics

Use this only if you specifically want the pondering branch worked out in serious detail.

```md
You previously proposed Budgeted Contingency Portfolio Pondering (BCPP) for Hydra. This time do not brainstorm new ideas. Turn BCPP into a code-grounded design and benchmark spec.

You are GPT-5.4 Pro operating in long-horizon research mode. Favor disciplined retrieval, explicit contracts, and benchmark-ready detail over broad ideation.

You are allowed to search widely for ingredients from contingency planning, algorithm portfolios, cache reuse, speculative execution, branch selection, adaptive submodularity, and test-time compute allocation. But your final answer must be specific enough to prototype in Hydra.

<output_contract>
- Return exactly the requested sections, in the requested order.
- Keep the answer compact and prototype-oriented.
</output_contract>

<research_mode>
- Work in 3 passes: unresolved questions, retrieval + second-order leads, final synthesis.
- Stop only when more searching is unlikely to improve the design materially.
</research_mode>

<completeness_contract>
- The answer is incomplete until contingency taxonomy, scheduler logic, reuse contract, benchmark, and kill criteria are all explicit.
</completeness_contract>

<verification_loop>
- Before finalizing, check whether a coding agent could implement the scheduler and reuse logic without guessing hidden contracts.
</verification_loop>

## What must be solved

The old answer says Hydra currently overweights probable-but-redundant futures and underweights lower-probability action-switching contingencies. Good. But that is still not enough to code.

You need to define:

1. the exact contingency event taxonomy
2. how candidate contingencies are generated cheaply
3. the exact feature definitions for `pi_j`, `V_j`, `r_j`, `Delta_place,j`, `type_j`, and any hardness score
4. the exact portfolio objective, budget accounting, and greedy/scheduler logic
5. how `PonderTask`, `PonderResult`, cache keys, and reuse selection would change
6. what nearest-branch reuse actually does at runtime
7. the cheapest benchmark that could falsify the idea quickly

## Working style

- Think deeply and search aggressively before deciding on a final mechanism.
- Search for both positive ingredients and failure cases from analogous systems.
- Keep the final answer narrow and buildable.

## Hydra grounding

Ground this in:

- `hydra-core/src/afbs.rs`
- `hydra-core/src/bridge.rs`
- current predictive pondering and cache/reuse surfaces
- current selective-compute doctrine in the docs

## Required output format

1. what current pondering likely misses
2. revised final BCPP mechanism
3. exact contingency representation and candidate-generation rules
4. scheduler / portfolio selector algorithm
5. runtime reuse contract
6. data structures and API sketch
7. benchmark and kill criteria
8. final verdict: prototype now / reserve shelf / reject

Success means the answer could drive a real prototype, not just another memo.
```

---

## Variant C — new open-ended invention prompt, but with code/testing pressure

Use this if you still want a broad invention turn, but you want it much deeper and more engineering-heavy than the first prompt-6 run.

```md
You are acting as a long-think breakthrough engineer for Hydra.

You are GPT-5.4 Pro operating in long-horizon research mode. Favor disciplined retrieval, evidence-backed synthesis, and auditable outputs over broad brainstorming.

Your job is to search broadly, combine good parts from different papers and systems, invent new Hydra-specific techniques where needed, and push them far enough toward implementation that a coding agent could plausibly prototype them.

Do not settle for a shallow invention memo. The previous run was too short and too vague.

<output_contract>
- Return exactly the requested sections, in the requested order.
- Keep the final answer compact, evidence-backed, and prototype-aware.
</output_contract>

<verbosity_controls>
- Prefer concise, information-dense writing.
- Do not omit mechanism, benchmark, or kill-criteria detail.
</verbosity_controls>

<research_mode>
- Work in 3 passes:
  1. Plan: identify the strongest unsolved Hydra bottlenecks and the most promising outside ingredient families.
  2. Retrieve: search broadly and follow 1-2 strong second-order leads for each serious candidate.
  3. Synthesize: keep only 1-3 serious techniques and reject weaker ideas explicitly.
- Stop only when more searching is unlikely to change the final ranking.
</research_mode>

<tool_persistence_rules>
- Search beyond the papers already surfaced when that could materially improve novelty or falsification.
- Do not stop at the first plausible invention.
</tool_persistence_rules>

<completeness_contract>
- Treat the task as incomplete until each surviving technique has mechanism, repo insertion point, prototype path, and kill criteria.
- Mark underspecified items [blocked] rather than pretending they are ready.
</completeness_contract>

<grounding_rules>
- Ground Hydra-specific claims in attached docs/code or raw links.
- Ground outside-technique claims in retrieved sources.
- Label inference as inference.
</grounding_rules>

<verification_loop>
- Before finalizing, ask whether each surviving technique is genuinely stronger than a generic bigger-model/bigger-search move.
- Check whether a coding agent could prototype the best candidate from your answer.
</verification_loop>

<dig_deeper_nudge>
- Do not stop at the first cool invention.
- Look for second-order risks, failure modes, and hidden dependencies.
- If the answer still feels like a neat memo instead of a prototypeable breakthrough, keep going.
</dig_deeper_nudge>

## Mission

Find the best 1-3 serious techniques that could create a real edge for Hydra through:

- better belief-search amortization
- better selective compute
- better idle-time reuse / pondering
- better strategic robustness under misspecification or opponent-style shift
- better decision-focused uncertainty handling

## How to work

- Explore widely before converging.
- Search outside Mahjong and outside the papers already surfaced if useful.
- Do second-order searching when a promising field or mechanism appears.
- Reject weak analogies hard.
- When you think you have a strong candidate, try to operationalize it:
  - what files would change?
  - what data structures would exist?
  - what benchmark would validate it?
  - what would kill it fast?

## Verification pressure

Before finalizing, pressure-test your own favorite ideas:

- what hidden assumption could break this?
- what tail case could make it worse than baseline?
- what cheap experiment would expose that quickly?
- if a coding agent tried to build this tomorrow, what would still be underspecified?

## Output requirements

Return only 1-3 serious techniques.

For each one, provide:

- problem solved
- borrowed ingredients and citations
- what is genuinely novel synthesis
- exact mechanism
- repo insertion points
- API / data-structure sketch if possible
- cheapest benchmark
- kill criteria
- what makes it plausibly stronger than the current mainline or reserve shelf

Then end with:

- the single best candidate to try first
- the single best cheap benchmark to run first
- the single biggest implementation risk

## Hard constraints

- no generic bigger-model or bigger-search proposals
- no AGPL borrowing
- no pretty but untestable inventions
- no keeping weak options alive just because they are novel

Success means your final answer feels like a 45-90 minute serious synthesis pass and leaves behind at least one technique that is both new and prototypeable.
```

---

## Recommended usage

- If you only spend one expensive turn on the old prompt-6 lane, use **Variant A**.
- If DEBC-AR still looks promising after that, use **Variant B** next.
- Use **Variant C** only if you want one more broad invention pass with stronger engineering pressure than the first run.
