# Hydra breakthrough prompt 6 — new technique inventor

Attach these files to the model:
- `hydra_breakthrough_docs_pack.zip`
- `deep_agent_20_pdfs.zip`

Zip structure the model should expect in `hydra_breakthrough_docs_pack.zip`:
- core design docs under `research/design/`
- runtime summary under `docs/`
- infra summary under `research/infrastructure/`
- prior answer archive under `research/agent_handoffs/prior_answers/`
- prompt files under `research/agent_handoffs/prompts/`

Zip structure the model should expect in `deep_agent_20_pdfs.zip`:
- paper PDFs under `deep_agent_20_pdfs/`

The attached zips should be sufficient for Hydra context, but for this task you are explicitly allowed to search and browse additional outside papers online if doing so helps invent better techniques. Use raw links only if the attachments are inaccessible or corrupted.

You are a research advisor trying to invent **new techniques** for Hydra — not just copy known ones — in the same spirit that LuckyJ’s strongest edges came from combining ideas into something strategically potent.

<memo_mode>
- Write in a compact invention memo style.
- Be explicit about what is borrowed, what is adapted, and what is newly proposed.
- Separate evidence-backed ingredients from genuinely novel synthesis.
</memo_mode>

<output_contract>
- Return exactly the requested sections, in order.
- Keep the answer compact and high-signal.
- Invent at most 3 serious techniques; fewer is better if the quality is higher.
</output_contract>

<verbosity_controls>
- Prefer concise, information-dense writing.
- Avoid broad brainstorming and weak option lists.
</verbosity_controls>

<research_mode>
- Work in 3 passes:
  1. Plan: identify the most promising ingredient ideas from Hydra and outside fields.
  2. Retrieve: gather enough evidence to understand those ingredients and their limits.
  3. Synthesize: invent 1-3 new mechanisms that fit Hydra and could matter for LuckyJ-level strength.
- Stop only when more searching is unlikely to materially improve the invented mechanisms.
</research_mode>

<citation_rules>
- Cite both Hydra docs and any outside evidence used as ingredients.
- Never fabricate references.
- Be clear when the final mechanism is novel synthesis rather than a direct paper transfer.
</citation_rules>

<grounding_rules>
- Every invented technique must have:
  1. a concrete Hydra insertion point,
  2. a clear problem it solves,
  3. a realistic cheap test,
  4. and an argument for why it could matter for beating LuckyJ.
- If an invented idea depends on a weak assumption, say so plainly.
</grounding_rules>

<completeness_contract>
- The task is incomplete until you have:
  1. invented 1-3 serious techniques,
  2. described their ingredients,
  3. given formulas / update rules / algorithm sketches,
  4. named insertion points and failure modes,
  5. proposed the cheapest meaningful experiments.
</completeness_contract>

<verification_loop>
- Before finalizing, check:
  - Is each technique actually new synthesis rather than a renamed known trick?
  - Is each one grounded enough to test?
  - Is each one aimed at a real Hydra bottleneck or leverage point?
</verification_loop>

<dig_deeper_nudge>
- Don’t just recombine ideas cosmetically.
- Look for combinations that change Hydra’s capabilities, not just its implementation details.
- Prefer techniques that create asymmetry: better selective compute, better exploitation of idle time, better uncertainty handling, better belief-search amortization, or better strategic robustness.
</dig_deeper_nudge>

Primary Hydra sources:
- `research/design/HYDRA_FINAL.md`
- `research/design/HYDRA_RECONCILIATION.md`
- `research/design/IMPLEMENTATION_ROADMAP.md`
- `research/design/OPPONENT_MODELING.md`
- `research/agent_handoffs/prior_answers/ANSWER_1-1.md`
- `research/agent_handoffs/prior_answers/ANSWER_2-1.md`
- `research/agent_handoffs/prior_answers/ANSWER_3-1.md`

Your job:
1. Search Hydra’s current ideas and outside ideas for ingredients.
2. Invent the **best 1–3 new techniques** Hydra could plausibly use to beat LuckyJ.
3. For each invented technique, provide:
   - name
   - problem solved
   - ingredient ideas it combines
   - exact mechanism
   - formulas / update rules / gating rules
   - insertion point in Hydra
   - failure modes
   - cheapest test
4. End with a ranking: if Hydra could only try one invented technique first, which one should it be?

Constraints:
- Do not invent fantasy mechanisms that cannot be tested cheaply.
- Do not propose broad architecture resets.
- Do not rely on AGPL code or implementation borrowing.
- Do not preserve weak inventions just because they sound novel.

Output format:
1. Best ingredient ideas
2. Invented techniques (1–3, ranked)
3. Technical specification for each
4. Cheap experiments
5. Kill criteria
6. Final pick

Success means you produce a small number of truly promising invented mechanisms that a coding agent could plausibly prototype inside Hydra.
