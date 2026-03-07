# Hydra long-think prompt pack 5 — cross-field follow-up and implementation expedition

This file contains multiple stronger prompt variants for the expensive external model that previously produced the shallow `prompt-5` style output.

Use the variant that matches how much budget you want to spend.

These variants are tuned for **GPT-5.4 Pro / long-think GPT-5.4-style models**:

- explicit output contracts
- strong completion criteria
- disciplined retrieval + synthesis flow
- anti-shallowness verification
- compact but evidence-rich final artifacts

---

## Shared attachment package for all variants

Attach these files if your UI allows multiple attachments:

- `hydra_breakthrough_docs_pack.zip`
- `hydra_breakthrough_thin_source_pack.zip`
- `deep_agent_20_pdfs.zip`

What each attachment contains:

- `hydra_breakthrough_docs_pack.zip`
  - core Hydra doctrine and design docs
  - prior answer archive
  - tracked handoff prompts
- `hydra_breakthrough_thin_source_pack.zip`
  - the docs pack plus thin source slices from `hydra-core` and `hydra-train`
  - enough code to ground engineering suggestions in the real repo
- `deep_agent_20_pdfs.zip`
  - local paper/reference PDFs already collected for Hydra

If the attachments are inaccessible, corrupted, or partially unreadable, use the raw GitHub links below with your browse tool.

### Raw GitHub fallback links

Core repo docs:

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

Thin source slices:

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

Reference links already surfaced as relevant to this track:

- Static and Dynamic Values of Computation in MCTS — https://proceedings.mlr.press/v124/sezener20a.html
- Distributionally Robust Neural Networks / group-shift robustness — https://iclr.cc/virtual/2020/poster/1491
- Conformal Risk Control — https://people.csail.mit.edu/tals/publication/conformal_risk/
- Monte-Carlo Graph Search / merging similar states — https://proceedings.mlr.press/v129/leurent20a.html
- Rao-Blackwellized Particle Filter review — https://asp-eurasipjournals.springeropen.com/articles/10.1155/2010/724087

---

## Variant A — focused follow-up on compute routing and robustness

Use this when you want the model to turn the best parts of the old cross-field answer into something a coding agent could actually build.

```md
You previously gave a useful but too-vague cross-field transfer memo for Hydra. This time do not give another broad survey.

You are GPT-5.4 Pro operating in a long-horizon research-and-engineering mode. Favor disciplined retrieval, evidence-backed synthesis, and strong completion behavior over fast ideation.

Your mission is to convert the two highest-value directions from that memo into a code-grounded engineering and experimentation brief:

1. compute routing / selective compute allocation
2. worst-group robust training over opponent-style or scenario slices

You are allowed to search widely across papers, official docs, GitHub repos, and related systems for better ingredients, but your final output must be grounded in Hydra's actual code and docs.

Do not over-constrain yourself early. Explore broadly first, then converge hard.

<output_contract>
- Return exactly the requested sections, in the requested order.
- Keep the final answer compact, high-signal, and implementation-oriented.
- Do not pad with generic background or repeat the prompt.
</output_contract>

<verbosity_controls>
- Prefer concise, information-dense writing.
- Keep intermediate framing brief.
- Do not shorten the answer so aggressively that formulas, interfaces, benchmarks, or failure checks become vague.
</verbosity_controls>

<tool_persistence_rules>
- Use browsing/search/retrieval aggressively when it materially improves correctness or novelty.
- Do not stop at the first plausible transfer or the first supporting paper.
- If one source seems promising, follow 1-2 second-order leads before finalizing.
- If a search path looks weak, abandon it explicitly and move on.
</tool_persistence_rules>

<dependency_checks>
- Before proposing implementation, verify Hydra already has or could cheaply expose the necessary signals, labels, and insertion points.
- Do not assume a data source, API surface, or training label exists until you verify it from the attachments or retrieved evidence.
</dependency_checks>

<research_mode>
- Work in 3 passes:
  1. Plan: identify 3-6 sub-questions that must be answered before a code-ready recommendation is possible.
  2. Retrieve: search each sub-question and follow 1-2 strong second-order leads.
  3. Synthesize: resolve contradictions, reject weak transfers, and write the final answer.
- Stop only when more searching is unlikely to change the final recommendation.
</research_mode>

<completeness_contract>
- Treat the task as incomplete until every requested deliverable is covered or explicitly marked [blocked].
- If one proposal remains underspecified, either finish specifying it or downgrade/reject it.
- Do not preserve a proposal just because it is interesting.
</completeness_contract>

<empty_result_recovery>
- If a search line comes back narrow or empty, try alternate wording, adjacent fields, or a stronger source before concluding there is no evidence.
</empty_result_recovery>

<citation_rules>
- Cite only sources you actually retrieved in this workflow or sources included in the attachments/raw links.
- Never fabricate references.
- Attach citations to the exact claims they support.
</citation_rules>

<grounding_rules>
- Base repo claims only on attached docs/code or retrieved raw links.
- If a statement is an inference rather than directly supported, label it as an inference.
- If sources conflict, state the conflict and resolve it explicitly.
</grounding_rules>

<verification_loop>
- Before finalizing, check:
  - did you explore more than one outside line of attack?
  - did you reject at least one plausible but weak transfer?
  - are file-level insertion points and required data/labels explicit?
  - is each proposal specific enough for a coding agent to start from?
</verification_loop>

<dig_deeper_nudge>
- Do not stop at the first plausible answer.
- Look for second-order issues, hidden dependencies, edge cases, and weak assumptions.
- If the answer would still read like a ranking memo, it is not done yet.
</dig_deeper_nudge>

## Working style

- Work in 5 passes:
  1. understand Hydra's current selective-compute and robustness surfaces from the attached docs/code
  2. search outside fields aggressively for strong ingredients and counterexamples
  3. reject weak analogies and keep only ideas that survive Hydra-specific scrutiny
  4. synthesize an implementation-ready spec for the top 1-2 ideas
  5. stress-test your own plan by trying to break it, then refine it
- Spend significant effort on verification and second-order searching before finalizing.
- If your answer would have been possible after only 20 minutes of thought, it is probably not deep enough.

## What counts as success

Do not stop until you have all of the following:

1. an exact compute-router proposal that could be implemented in Hydra without guessing core interfaces
2. an exact worst-group or minimax-robust training proposal that could be implemented on top of current Hydra heads/data
3. file-level insertion points in the real repo
4. algorithm sketches or pseudocode with enough detail that a coding agent could start work
5. a concrete benchmark and kill-criteria protocol for each proposal
6. a section explaining why these ideas are stronger than simply making Hydra bigger or more complex

## Strong guidance

- Search outside Mahjong. Good fields include metareasoning, test-time compute allocation, algorithm portfolios, selective prediction, robust optimization, multi-source minimax, calibration under shift, and risk-sensitive decision systems.
- You may discover that a different outside idea is stronger than the original compute-router / worst-group pair. If so, say that clearly — but only if you can defend it at a higher standard.
- You are allowed to synthesize new variants or hybrids if the papers alone are not enough.
- Keep the final recommendation small. One or two serious routes beat a laundry list.

## Hydra-specific grounding requirements

Ground every serious proposal in:

- exact files/functions/structs likely to change
- what data already exists and what new logging/labels are required
- what can be implemented now vs what would need later infrastructure
- how it interacts with current AFBS / Hand-EV / belief / search-trust / danger / tenpai surfaces

## Required output format

1. what Hydra already has that matters for this question
2. strongest outside ingredients found
3. final proposal A — compute routing or stronger replacement
   - problem solved
   - borrowed ingredients
   - exact mechanism
   - Hydra insertion points
   - needed labels / logs / supervision
   - pseudocode or algorithm loop
   - benchmark and kill criteria
4. final proposal B — worst-group robustness or stronger replacement
   - same sub-sections as above
5. what you explicitly rejected and why
6. final recommendation: what Hydra should actually try first

## Hard constraints

- no generic "use transformers", "use more search", or "just do bigger RL"
- no AGPL-derived code advice
- no shallow ranking memo
- no pretending a proposal is code-ready if key interfaces or labels are still unknown

Success means your output is specific enough that a coding agent could immediately translate it into a real work plan with minimal guesswork.
```

---

## Variant B — new open-ended prompt for cross-field breakthrough to prototype

Use this when you want the model to range more freely than Variant A, while still forcing code and testing discipline.

```md
You are acting as a long-think breakthrough engineer for Hydra, a Riichi Mahjong AI that already has a strong Rust codebase, belief/search machinery, and a reconciled execution doctrine.

You are GPT-5.4 Pro operating in long-horizon research mode. Favor disciplined retrieval, sharp synthesis, and explicit completion criteria over broad brainstorming.

Your job is not to write a generic research memo. Your job is to search broadly, synthesize aggressively, and come back with the smallest number of cross-field imports that could plausibly create a real edge over LuckyJ.

<output_contract>
- Return exactly the requested sections, in the requested order.
- Keep the answer compact, evidence-backed, and prototype-oriented.
</output_contract>

<verbosity_controls>
- Prefer concise, information-dense writing.
- Avoid generic scene-setting.
- Preserve enough detail for code and benchmark implications.
</verbosity_controls>

<research_mode>
- Work in 3 passes:
  1. Plan: identify the most promising outside fields and Hydra bottlenecks to connect.
  2. Retrieve: search broadly, then follow 1-2 strong second-order leads for each serious candidate.
  3. Synthesize: reject weak mappings and keep only 1-3 serious candidates.
- Stop only when more searching is unlikely to change the final ranking.
</research_mode>

<tool_persistence_rules>
- Search outside Mahjong aggressively.
- Do not stop at the first adjacent paper.
- Use additional retrieval when it materially improves novelty, grounding, or falsification.
</tool_persistence_rules>

<completeness_contract>
- Treat the task as incomplete until each surviving candidate includes mechanism, repo insertion point, prototype path, and kill criteria.
- Mark any underspecified item [blocked] rather than pretending it is ready.
</completeness_contract>

<grounding_rules>
- Ground Hydra-specific claims in the attachments/raw links.
- Ground outside-technique claims in retrieved sources.
- Label inference as inference.
</grounding_rules>

<verification_loop>
- Before finalizing, verify that each surviving candidate is stronger than a generic bigger-model or bigger-search move.
- Check whether a coding agent could actually prototype it from your writeup.
</verification_loop>

<dig_deeper_nudge>
- Do not stop at the first cool transfer.
- Prefer techniques that create asymmetry, not cosmetic complexity.
</dig_deeper_nudge>

Hydra is most interested in techniques that create asymmetry through:

- better selective compute
- better reuse of idle time / pondering
- better robustness under opponent-style shift and belief misspecification
- better calibration and risk control
- better belief-search amortization
- better exploitation of existing Hydra surfaces instead of architecture resets

## What to do

1. Read the attached docs and thin source slices hard enough to understand what Hydra really already has.
2. Search outside fields and remote repos aggressively. Do not stop at the first adjacent paper.
3. Combine good parts from multiple papers when that produces something stronger than any single transfer.
4. For the top serious candidates, go beyond strategy: sketch the mechanism, the code shape, and the validation path.
5. If a candidate seems promising, push it one step further and try to imagine how you would prototype it in the actual Hydra files.
6. Try to break your own best ideas before finalizing.

## Exploration rules

- Do not constrain yourself to the fields named in the old prompt-5 output.
- You may search any field with structural similarity to Hydra's problems.
- You may propose a new hybrid technique if it is genuinely better than a direct transfer.
- However, your final answer must still be concrete enough for coding and experimentation.

## Deliverables

Return only 1-3 serious candidates.

For each candidate give:

- name
- outside ingredients and sources
- why it transfers to Hydra specifically
- exact Hydra surfaces it would touch
- implementation sketch or pseudocode
- cheapest prototype path
- what success would look like
- what would kill the idea quickly

Then end with:

- the single best candidate to try first
- the single best cheap benchmark to run first
- the single biggest hidden risk in your recommendation

## Important tone / depth guidance

- This should feel like the output of a 45-90 minute serious research pass, not a fast brainstorm.
- Prefer deep verification, rejection of weak ideas, and strong synthesis over breadth.
- If attachments fail, use the raw links provided.
```

---

## Recommended usage

- If you only spend one expensive turn on the old prompt-5 lane, use **Variant A**.
- If you want a broader but still serious rethink from that lane, use **Variant B**.
