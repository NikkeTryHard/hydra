# Hydra fresh-context prompt — cross-field breakthrough to prototype

Attach these files to the model:
- `hydra_breakthrough_docs_pack.zip`
- `hydra_breakthrough_thin_source_pack.zip`
- `deep_agent_20_pdfs.zip` if available

If `deep_agent_20_pdfs.zip` is not available, proceed anyway using the two Hydra zip packs plus online retrieval.

The attached zips should be sufficient for Hydra context, but for the **core architecture docs** you should prefer raw GitHub links with your browse/fetch tool whenever your environment tends to fragment long files. Use the zips for structure, file discovery, and supplemental grounding.

## Critical directive — how to read the core Hydra docs

You must avoid a known bad behavior: fragmented keyword-peeking over large architecture docs.

Bad behavior for this task:
- searching for keywords first
- reading isolated 20-100 line chunks around those keywords
- treating the docs like logs or a grep database
- inventing new directions before understanding Hydra as a whole system

For this task, that behavior is disqualifying.

Required reading workflow:
1. Use your **browse/fetch tool on the raw GitHub links** for the core docs listed below.
2. Read those core docs **holistically and sequentially** before doing narrower searching.
3. Build a high-level model of what Hydra already is, what is active, what is reserve, and what loops are already partially closed.
4. Only after that may you use narrower searching for exact details and outside analogies.

Do **not** use grep-style keyword hunting as your primary reading strategy for these core docs.

Core docs that must be read holistically first:
- `research/design/HYDRA_RECONCILIATION.md`
- `research/design/HYDRA_FINAL.md`
- `docs/GAME_ENGINE.md`
- `research/design/OPPONENT_MODELING.md`
- `research/agent_handoffs/prior_answers/ANSWER_1-1.md`
- `research/agent_handoffs/prior_answers/ANSWER_2-1.md`
- `research/agent_handoffs/prior_answers/ANSWER_3-1.md`

If the zip attachment is easy for you to read holistically, you may use it. But if your environment tends to fragment large files or only surface narrow line windows, prefer the **raw GitHub links with the browse/fetch tool** for the core docs above.

<holistic_ingestion_rules>
- Read the core docs as whole documents before narrowing.
- Do not start with keyword search on the core docs.
- Do not rely on fragmented line-window retrieval for architecture understanding.
- After holistic reading, you may use targeted search for exact details.
</holistic_ingestion_rules>

## Attachment strategy for this prompt

This is a **fresh-context** run. The model should not assume any prior Hydra memory.

Recommended attachment set:
- `hydra_breakthrough_docs_pack.zip`
- `hydra_breakthrough_thin_source_pack.zip`
- `deep_agent_20_pdfs.zip`

For fresh agents, attach all 3 if possible.

## Zip contents — read this first instead of wandering

### `hydra_breakthrough_docs_pack.zip`

```text
research/design/
  HYDRA_FINAL.md
  HYDRA_RECONCILIATION.md
  IMPLEMENTATION_ROADMAP.md
  OPPONENT_MODELING.md
  TESTING.md
  SEEDING.md
  AGENT_FOLLOWUP_1.md
  AGENT_FOLLOWUP_2.md
  AGENT_FOLLOWUP_3.md
research/infrastructure/
  INFRASTRUCTURE.md
research/agent_handoffs/
  HANDOFF_PACKAGE_GUIDE.md
  prior_answers/
  prompts/
research/
  BUILD_AGENT_PROMPT.md
docs/
  GAME_ENGINE.md
```

Start here:
1. `research/design/HYDRA_RECONCILIATION.md`
2. `research/design/HYDRA_FINAL.md`
3. `docs/GAME_ENGINE.md`
4. `research/design/OPPONENT_MODELING.md`
5. `research/agent_handoffs/prior_answers/ANSWER_1-1.md`
6. `research/agent_handoffs/prior_answers/ANSWER_2-1.md`
7. `research/agent_handoffs/prior_answers/ANSWER_3-1.md`

### `hydra_breakthrough_thin_source_pack.zip`

Includes everything in the docs pack, plus:

```text
hydra-core/src/
  encoder.rs
  ct_smc.rs
  afbs.rs
  hand_ev.rs
  endgame.rs
  robust_opponent.rs
hydra-train/src/
  model.rs
  data/sample.rs
  data/mjai_loader.rs
  training/losses.rs
```

Use this mainly to anchor proposals in real file/function surfaces.

### `deep_agent_20_pdfs.zip`

Paper/reference bundle. Use this after you understand Hydra, not before.

## Raw GitHub fallback links

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
- `research/agent_handoffs/prior_answers/ANSWER_1-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_1-1.md
- `research/agent_handoffs/prior_answers/ANSWER_2-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_2-1.md
- `research/agent_handoffs/prior_answers/ANSWER_3-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_3-1.md

You are acting as a long-think breakthrough engineer for Hydra, a Riichi Mahjong AI whose goal is to reach or exceed LuckyJ-level strength.

You are GPT-5.4 Pro operating in long-horizon research mode. Favor disciplined retrieval, evidence-backed synthesis, and explicit completion behavior over broad brainstorming.

Your job is not to write a generic research memo. Your job is to search broadly, synthesize aggressively, and come back with the smallest number of cross-field imports that could plausibly create a real edge over LuckyJ and are concrete enough for later coding and testing.

Hydra is most interested in techniques that create asymmetry through:
- better selective compute
- better reuse of idle time / pondering
- better robustness under opponent-style shift and belief misspecification
- better calibration and risk control
- better belief-search amortization
- better exploitation of existing Hydra surfaces instead of architecture resets

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
  1. Ingest: read the core Hydra docs holistically first using the raw links or equivalent full-document browsing.
  2. Retrieve: identify the most promising outside fields and Hydra bottlenecks to connect, then follow 1-2 strong second-order leads for each serious candidate.
  3. Synthesize: reject weak mappings and keep only 1-3 serious candidates.
- Stop only when more searching is unlikely to change the final ranking.
</research_mode>

<tool_persistence_rules>
- Prefer full-document browse/fetch for core docs over fragmented terminal-style chunk reading.
- Search outside Mahjong aggressively.
- Do not stop at the first adjacent paper.
- Use additional retrieval when it materially improves novelty, grounding, or falsification.
</tool_persistence_rules>

<dependency_checks>
- Before proposing implementation, verify Hydra already has or could cheaply expose the needed signals, labels, or runtime hooks.
</dependency_checks>

<completeness_contract>
- Treat the task as incomplete until each surviving candidate includes mechanism, repo insertion point, prototype path, benchmark, and kill criteria.
- Mark any underspecified item [blocked] rather than pretending it is ready.
</completeness_contract>

<citation_rules>
- Cite only sources you actually retrieved in this workflow or sources included in the attachments/raw links.
- Never fabricate references.
- Attach citations to the exact claims they support.
</citation_rules>

<grounding_rules>
- Ground Hydra-specific claims in the attachments/raw links.
- Ground outside-technique claims in retrieved sources.
- Label inference as inference.
</grounding_rules>

<verification_loop>
- Before finalizing, verify that you actually read the core Hydra docs holistically before narrowing in.
- Before finalizing, verify that each surviving candidate is stronger than a generic bigger-model or bigger-search move.
- Check whether a coding agent could actually prototype it from your writeup.
</verification_loop>

<dig_deeper_nudge>
- Do not stop at the first cool transfer.
- Prefer techniques that create asymmetry, not cosmetic complexity.
</dig_deeper_nudge>

## What to do

1. Read the attached docs and thin source slices hard enough to understand what Hydra really already has.
2. Search outside fields and remote repos aggressively. Do not stop at the first adjacent paper.
3. Combine good parts from multiple papers when that produces something stronger than any single transfer.
4. For the top serious candidates, go beyond strategy: sketch the mechanism, the code shape, and the validation path.
5. If a candidate seems promising, push it one step further and try to imagine how you would prototype it in the actual Hydra files.
6. Try to break your own best ideas before finalizing.

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

## Hard constraints

- no generic bigger-model or bigger-search proposals
- no AGPL-derived code advice
- no broad architecture resets
- no pretty but untestable ideas
- no keeping weak options alive just because they are novel

Success means your final answer feels like a serious long-think synthesis pass and leaves behind at least one technique that is both high-upside and prototypeable.
