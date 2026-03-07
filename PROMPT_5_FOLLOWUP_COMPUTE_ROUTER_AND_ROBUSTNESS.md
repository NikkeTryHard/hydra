# Hydra follow-up for agent 5 — compute routing and worst-group robustness

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
- building recommendations from scattered snippets instead of whole-document understanding

For this task, that behavior is disqualifying.

Required reading workflow:
1. Use your **browse/fetch tool on the raw GitHub links** for the core docs listed below.
2. Read those core docs **holistically and sequentially** before doing narrower searching.
3. Build a high-level mental model of how Hydra's modules and priorities fit together.
4. Only after that may you use narrower searching to answer detailed implementation questions.

Do **not** use grep-style keyword hunting as your primary reading strategy for these core docs.

Core docs that must be read holistically first:
- `research/design/HYDRA_RECONCILIATION.md`
- `research/design/HYDRA_FINAL.md`
- `docs/GAME_ENGINE.md`
- `research/agent_handoffs/prior_answers/ANSWER_1-1.md`
- `research/agent_handoffs/prior_answers/ANSWER_2-1.md`
- `research/agent_handoffs/prior_answers/ANSWER_3-1.md`

If the zip attachment is easy for you to read holistically, you may use it. But if your environment tends to fragment large files or only surface narrow line windows, prefer the **raw GitHub links with the browse/fetch tool** for the core docs above.

Only after the core docs are ingested holistically may you narrow in on:
- code insertion points
- supporting design docs
- exact file/function references
- outside papers and adjacent fields

<holistic_ingestion_rules>
- Read the core docs as whole documents before narrowing.
- Do not start with keyword search on the core docs.
- Do not rely on fragmented line-window retrieval for architecture understanding.
- After holistic reading, you may use targeted search for exact details.
</holistic_ingestion_rules>

## Attachment strategy for this prompt

This is a **follow-up** prompt for agent 5, not a fresh-context run.

- If the model already has the earlier Hydra context from the prior prompt-5 run, the minimum good attachment set is:
  - `hydra_breakthrough_docs_pack.zip`
  - `deep_agent_20_pdfs.zip`
- If you want maximum grounding and less guessing about code insertion points, also attach:
  - `hydra_breakthrough_thin_source_pack.zip`

Recommended for best results: attach all 3 if the UI allows it.

## Zip contents — read this first instead of wandering

### `hydra_breakthrough_docs_pack.zip`

Primary docs and prior-answer tree:

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
    ANSWER_1.md
    ANSWER_1-1.md
    ANSWER_2.md
    ANSWER_2-1.md
    ANSWER_3.md
    ANSWER_3-1.md
  prompts/
    PROMPT_1_TECHNICAL_BREAKTHROUGH.md
    PROMPT_2_REPO_AWARE_NEXT_TRANCHE.md
    PROMPT_3_STRATEGIC_CUTTER.md
    PROMPT_4_OUTSIDE_THE_BOX_BUT_GROUNDED.md
    PROMPT_5_CROSS_FIELD_TRANSFER_HUNTER.md
    PROMPT_6_NEW_TECHNIQUE_INVENTOR.md
research/
  BUILD_AGENT_PROMPT.md
docs/
  GAME_ENGINE.md
```

Read order inside this zip:
1. `research/design/HYDRA_RECONCILIATION.md`
2. `research/design/HYDRA_FINAL.md`
3. `docs/GAME_ENGINE.md`
4. `research/agent_handoffs/prior_answers/ANSWER_1-1.md`
5. `research/agent_handoffs/prior_answers/ANSWER_2-1.md`
6. `research/agent_handoffs/prior_answers/ANSWER_3-1.md`

### `hydra_breakthrough_thin_source_pack.zip`

This includes everything in the docs pack, plus thin code slices:

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

Use this zip mainly for:
- file/function insertion points
- checking whether needed signals already exist
- avoiding fake implementation claims

### `deep_agent_20_pdfs.zip`

Paper/reference bundle. Use it for cross-field evidence and second-order searching, not for discovering Hydra repo structure.

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

Prior answer anchors:
- `research/agent_handoffs/prior_answers/ANSWER_1-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_1-1.md
- `research/agent_handoffs/prior_answers/ANSWER_2-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_2-1.md
- `research/agent_handoffs/prior_answers/ANSWER_3-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_3-1.md

Reference links already surfaced as relevant:
- Static and Dynamic Values of Computation in MCTS — https://proceedings.mlr.press/v124/sezener20a.html
- Distributionally Robust Neural Networks / group shift — https://iclr.cc/virtual/2020/poster/1491
- Conformal Risk Control — https://people.csail.mit.edu/tals/publication/conformal_risk/
- Monte-Carlo Graph Search — https://proceedings.mlr.press/v129/leurent20a.html
- Rao-Blackwellized Particle Filter review — https://asp-eurasipjournals.springeropen.com/articles/10.1155/2010/724087

You previously gave a useful but too-vague cross-field transfer memo for Hydra. This time do not give another broad survey.

You are GPT-5.4 Pro operating in a long-horizon research-and-engineering mode. Favor disciplined retrieval, evidence-backed synthesis, and strong completion behavior over fast ideation.

Your mission is to convert the two highest-value directions from that memo into a code-grounded engineering and experimentation brief:

1. compute routing / selective compute allocation
2. worst-group robust training over opponent-style or scenario slices

You are allowed to search widely across papers, official docs, GitHub repos, and related systems for better ingredients, better objectives, better interfaces, and stronger benchmark design, but your final output must be grounded in Hydra's actual code and docs.

Do not over-constrain yourself early. Explore broadly first, then converge hard.

<output_contract>
- Return exactly the requested sections, in the requested order.
- Keep the final answer compact, high-signal, and implementation-oriented.
- Do not pad with generic background or repeat the prompt.
</output_contract>

<verbosity_controls>
- Prefer concise, information-dense writing.
- Keep framing brief.
- Do not shorten the answer so aggressively that formulas, interfaces, benchmarks, or failure checks become vague.
</verbosity_controls>

<research_mode>
- Work in 3 passes:
  1. Ingest: read the core Hydra docs holistically first using the raw links or equivalent full-document browsing.
  2. Retrieve: identify 3-6 sub-questions, search each one, and follow 1-2 strong second-order leads.
  3. Synthesize: resolve contradictions, reject weak transfers, and write the final answer.
- Stop only when more searching is unlikely to change the final recommendation.
</research_mode>

<tool_persistence_rules>
- Use browsing/search/retrieval aggressively when it materially improves correctness or novelty.
- Prefer full-document browse/fetch for core docs over fragmented terminal-style chunk reading.
- Do not stop at the first plausible transfer or the first supporting paper.
- If one source seems promising, follow 1-2 second-order leads before finalizing.
- If a search path looks weak, abandon it explicitly and move on.
</tool_persistence_rules>

<dependency_checks>
- Before proposing implementation, verify Hydra already has or could cheaply expose the necessary signals, labels, and insertion points.
- Do not assume a data source, API surface, or training label exists until you verify it from the attachments or retrieved evidence.
</dependency_checks>

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
  - did you actually read the core Hydra docs holistically before narrowing in?
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
