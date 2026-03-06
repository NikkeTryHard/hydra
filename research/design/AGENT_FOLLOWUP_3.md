# Hydra deep-agent follow-up for ANSWER_3-style agent

I have attached:
- `hydra_agent_handoff_docs_only.zip` — this is the primary source material
- `deep_agent_20_pdfs` — the primary paper/reference package

Use the docs zip first. If you cannot access the zip, use the raw GitHub markdown links I provide separately. Do NOT rely on normal GitHub browsing/search or generic web search to reconstruct the project context.

Your job in this prompt is NOT to inspect source code and NOT to write or integrate code. Your job is to look at Hydra’s CURRENT PLAN and make it:
1. stronger,
2. more coherent,
3. more likely to produce real breakthroughs in Mahjong AI,
4. still grounded enough to be implementable by a separate coding agent later.

I do NOT want generic brainstorming. I want a hard, evidence-backed **second pruning / prioritization pass** on the plan *after* several key strategic questions have already been settled.

You should read the docs as a coherent program, especially:
- `research/design/HYDRA_FINAL.md`
- `research/design/HYDRA_RECONCILIATION.md`
- `research/design/IMPLEMENTATION_ROADMAP.md`
- `research/BUILD_AGENT_PROMPT.md`
- `research/design/OPPONENT_MODELING.md`
- `research/design/TESTING.md`
- `research/infrastructure/INFRASTRUCTURE.md`
- `docs/GAME_ENGINE.md`
- `research/design/SEEDING.md`
- plus any evidence/comparison docs that materially affect your conclusions

Treat `research/BUILD_AGENT_PROMPT.md` as an execution-rigor overlay on the design docs, but your task here is purely research/design.

## Settled decisions you should treat as fixed inputs

Do **not** spend time re-arguing these unless you think one is clearly catastrophic:

- Hydra should **not** restart from scratch.
- Hydra’s active path is **supervision-first before search-expansion-first**.
- Use **one unified belief stack**: Mixture-SIB + CT-SMC; no duplicate standalone belief machinery.
- **Hand-EV comes before deeper AFBS**.
- **AFBS is selective/specialist**, not broad default runtime.
- **DRDA/ACH moves off the critical path** and lives on the reserve shelf.
- **Oracle guidance should be aligned** so the teacher stays teachable.
- Broad “search everywhere,” duplicated belief stacks, and early optimizer-theory detours are not on the active path.

External evidence also already supports several pattern-level choices here:
- unified public-belief-style state abstractions are real, not made-up Hydra weirdness
- aligned oracle/teacher guidance is more defensible than unconstrained privileged distillation
- robustness belongs in the core solving/objective layer, though Hydra-specific placement details still require judgment

You do not need to spend time re-proving those pattern-level claims unless you have a very strong contrary argument.

## What I want you to do

I want you to critique the **current reconciled Hydra plan** and make one more hard call on what remains:

### Part 1 — Critique the reconciled active path
Identify where the reconciled active Hydra path is still:
- too fragile,
- too underspecified,
- too compute-inefficient,
- too likely to stall before real strength,
- or still carrying too much reserve-shelf baggage in disguised form.

### Part 2 — Re-rank the reserve shelf
The reconciliation memo keeps several old good ideas on the reserve shelf. I want you to sort them harder:

- which reserve ideas are actually strong **phase-next** candidates,
- which are long-shot but worth preserving,
- which should probably be demoted even further,
- and which one or two reserve ideas have the best “if active Hydra underdelivers, try this next” upside.

Focus especially on:
- robust-opponent search backups vs confidence-gated safe exploitation
- richer latent opponent posterior
- deeper AFBS semantics
- stronger endgame exactification
- incremental/structured belief updates
- any remaining optimizer/game-theory ideas

### Part 3 — Identify the strongest breakthrough bets that still survive the pruning
Do not just list cool ideas. I want the **best surviving breakthrough bets after the active-path cuts have already happened**.

For each surviving bet, tell me:
- why it is still alive after pruning,
- why it might matter specifically in Mahjong,
- what evidence supports it,
- what assumption it relies on,
- why it might still fail,
- and the cheapest meaningful experiment to test it.

### Part 4 — Fill in the remaining strategic blanks
For any reserve or breakthrough idea you keep alive, fill in the missing technical details that the docs still leave abstract:
- formulas,
- objective functions,
- update rules,
- gating criteria,
- thresholds,
- approximate algorithms,
- calibration procedures,
- evaluation metrics,
- stopping rules.

## Constraints

- Do NOT inspect source code.
- Do NOT pretend you implemented or validated anything.
- Do NOT give broad generic summaries of Mahjong AI history unless directly relevant.
- Do NOT recommend things that obviously blow up latency/compute without addressing feasibility.
- Do NOT rely on AGPL code or implementation borrowing.
- Keep proposals compatible with a separate coding agent implementing them later.

Assume a separate coding agent will take your response and use it as the strategic decision layer above concrete implementation work.

## How to reason about evidence

Use a strict evidence hierarchy:
1. direct Mahjong evidence,
2. direct imperfect-information game AI evidence,
3. adjacent multiplayer/search/belief modeling evidence,
4. cross-disciplinary evidence that transfers unusually well.

When evidence is weak, say so clearly.
When an idea is speculative, quantify that.
When you think something is novel but unproven, separate that from evidence-backed recommendations.

## Required output format

Give me the answer in this structure:

### 1. Executive verdict
- Is the reconciled active Hydra plan strong enough to pursue as the mainline?
- Where is it strongest?
- Where is it still most likely to fail?
- Which reserve-shelf idea is most likely to matter next if mainline underdelivers?

### 2. Hardest remaining weaknesses in the reconciled active path
Rank the most important remaining weaknesses or blind spots.

### 3. Re-ranked reserve shelf
For each major reserve idea:
- keep / demote / mostly drop
- why
- evidence basis
- feasibility
- upside
- risk

### 4. Best surviving breakthrough bets
Give me your best 3–5 surviving high-upside ideas.
For each:
- novelty level
- evidence level
- why it could matter specifically in Mahjong
- minimum viable experiment
- what success would look like

### 5. Concrete technical fill-ins
Where the remaining reserve/breakthrough ideas are too vague, provide:
- formulas
- losses
- update rules
- thresholds
- hyperparameters
- pseudocode / compact illustrative snippets
- interface/data-flow guidance for a coding agent

### 6. Recommended revised research agenda after reconciliation
Give me a revised prioritized roadmap:
- active mainline must-have
- phase-next multipliers
- reserve shelf worth preserving
- likely dead ends / no-longer-worth-it complexity

### 7. Evaluation plan
Tell me how to know whether the pruned/revised plan is actually better:
- ablations
- matchups
- metrics
- failure signals
- stopping criteria

### 8. Final recommendation
If you had to reshape Hydra into the strongest and most coherent feasible version of itself **after this reconciliation pass**, what exact direction would you choose and why?

## Style requirements

I want:
- depth,
- specificity,
- formulas,
- precise proposals,
- evidence-backed reasoning,
- clear separation between proven, plausible, and speculative ideas.

I do NOT want:
- fluff,
- generic “future work,”
- shallow novelty for novelty’s sake,
- or re-arguing already-settled choices unless you have a genuinely strong reason.

Your goal is to help make Hydra not just more complete, but more formidable and more coherent after the active-path / reserve-shelf split has already been made.
