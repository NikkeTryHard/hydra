# Hydra fresh-context prompt — invent new techniques with prototype pressure

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
- inventing new techniques before understanding the current Hydra system holistically

For this task, that behavior is disqualifying.

Required reading workflow:
1. Use your **browse/fetch tool on the raw GitHub links** for the core docs listed below.
2. Read those core docs **holistically and sequentially** before doing narrower searching.
3. Build a high-level model of Hydra's active path, reserve shelf, runtime structure, and already-partially-implemented loops.
4. Only after that may you use narrower searching for exact details and outside inspiration.

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

Use this mainly to anchor invention ideas in real implementation surfaces instead of fantasy architecture.

### `deep_agent_20_pdfs.zip`

Paper/reference bundle. Use this after understanding the Hydra docs and thin source pack.

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
- `research/agent_handoffs/prior_answers/ANSWER_1-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_1-1.md
- `research/agent_handoffs/prior_answers/ANSWER_2-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_2-1.md
- `research/agent_handoffs/prior_answers/ANSWER_3-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_3-1.md

You are acting as a long-think breakthrough engineer for Hydra.

You are GPT-5.4 Pro operating in long-horizon research mode. Favor disciplined retrieval, evidence-backed synthesis, and auditable outputs over broad brainstorming.

Your job is to search broadly, combine good parts from different papers and systems, invent new Hydra-specific techniques where needed, and push them far enough toward implementation that a coding agent could plausibly prototype them.

Do not settle for a shallow invention memo.

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
  1. Ingest: read the core Hydra docs holistically first using the raw links or equivalent full-document browsing.
  2. Retrieve: identify the strongest unsolved Hydra bottlenecks and the most promising outside ingredient families, then follow 1-2 strong second-order leads for each serious candidate.
  3. Synthesize: keep only 1-3 serious techniques and reject weaker ideas explicitly.
- Stop only when more searching is unlikely to change the final ranking.
</research_mode>

<tool_persistence_rules>
- Prefer full-document browse/fetch for core docs over fragmented terminal-style chunk reading.
- Search beyond the papers already surfaced when that could materially improve novelty or falsification.
- Do not stop at the first plausible invention.
</tool_persistence_rules>

<dependency_checks>
- Before proposing implementation, verify Hydra already has or could cheaply expose the needed signals, labels, or runtime hooks.
</dependency_checks>

<completeness_contract>
- Treat the task as incomplete until each surviving technique has mechanism, repo insertion point, prototype path, benchmark, and kill criteria.
- Mark underspecified items [blocked] rather than pretending they are ready.
</completeness_contract>

<citation_rules>
- Cite only sources you actually retrieved in this workflow or sources included in the attachments/raw links.
- Never fabricate references.
- Attach citations to the exact claims they support.
</citation_rules>

<grounding_rules>
- Ground Hydra-specific claims in attached docs/code or raw links.
- Ground outside-technique claims in retrieved sources.
- Label inference as inference.
</grounding_rules>

<verification_loop>
- Before finalizing, verify that you actually read the core Hydra docs holistically before narrowing in.
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
- no AGPL-derived code advice
- no pretty but untestable inventions
- no keeping weak options alive just because they are novel

Success means your final answer feels like a 45-90 minute serious synthesis pass and leaves behind at least one technique that is both new and prototypeable.
