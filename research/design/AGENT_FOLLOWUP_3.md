# Hydra deep-agent follow-up for ANSWER_3-style agent

I have attached:
- `hydra_agent_handoff_docs_only.zip` — the primary docs package
- `deep_agent_20_pdfs.zip` — the primary paper/reference package

Use the docs zip first. If you cannot access it, use the raw markdown links I provide separately. Do **not** rely on normal GitHub browsing/search or generic web search to reconstruct the project context.

Your job in this prompt is **not** to inspect source code and **not** to write or integrate code. Your job is to take Hydra’s **current reconciled plan** and perform a second hard strategic pass that makes it:
1. stronger,
2. cleaner,
3. more likely to produce real Mahjong strength,
4. and more disciplined about what belongs on the active path versus the reserve shelf.

I do **not** want generic brainstorming. I want a hard, evidence-backed **post-reconciliation pruning and prioritization pass**.

## Read these first

Treat these as the main input stack:
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
- Hydra should use **one unified belief stack**: Mixture-SIB + CT-SMC; no duplicated standalone belief machinery.
- **Hand-EV comes before deeper AFBS**.
- **AFBS is selective / specialist / hard-state-gated**, not broad default runtime.
- **DRDA/ACH moves off the critical path** and lives on the reserve shelf.
- **Oracle guidance should be aligned** so the teacher stays teachable.
- Broad “search everywhere,” duplicated belief stacks, and early optimizer-theory detours are not on the active path.

You should also assume these pattern-level claims are already sufficiently supported by external evidence:
- unified public-belief-style state abstractions are real, not made-up Hydra weirdness
- aligned oracle/teacher guidance is more defensible than unconstrained privileged distillation
- robustness belongs in the core solving/objective layer, even if Hydra-specific placement details still require judgment

Do **not** waste your budget re-proving those pattern-level points unless you have a very strong contrary argument.

## What I want you to do

I want one more hard call on the **reconciled Hydra plan**.

### Part 1 — Pressure-test the active path
Tell me where the reconciled active Hydra plan is still:
- too fragile,
- too underspecified,
- too compute-inefficient,
- too likely to stall before real strength,
- or still quietly carrying reserve-shelf ideas that should not really be there.

### Part 2 — Re-rank the reserve shelf harder
`HYDRA_RECONCILIATION.md` keeps several ideas alive on the reserve shelf. Sort them more aggressively.

For each major reserve idea, decide whether it is:
- **phase-next**,
- **worth preserving but not near-term**,
- **mostly demote**,
- or **probably not worth the complexity**.

Focus especially on:
- robust-opponent search backups vs confidence-gated safe exploitation
- richer latent opponent posterior
- deeper AFBS semantics
- stronger endgame exactification
- incremental / structured belief updates
- any remaining optimizer / game-theory ideas

### Part 3 — Keep only the best surviving breakthrough bets
I do **not** want a long list. I want the strongest **3–5 surviving breakthrough bets after pruning**.

For each one, tell me:
- why it survives the pruning,
- why it could matter specifically in Mahjong,
- what evidence supports it,
- what assumption it relies on,
- why it might still fail,
- and the cheapest meaningful experiment to test it.

### Part 4 — Fill in the remaining strategic blanks
For any reserve or breakthrough idea you keep alive, fill in the missing technical details that are still too abstract:
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

- Do **not** inspect source code.
- Do **not** pretend you implemented or validated anything.
- Do **not** give broad generic summaries of Mahjong AI history unless directly relevant.
- Do **not** recommend things that obviously blow up latency/compute without addressing feasibility.
- Do **not** rely on AGPL code or implementation borrowing.
- Keep proposals compatible with a separate coding agent implementing them later.

Assume a separate coding agent will use your answer as the strategic decision layer above concrete implementation work.

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

Give me the answer in this exact structure:

### 1. Executive verdict
- Is the reconciled active Hydra plan strong enough to pursue as the mainline?
- Where is it strongest?
- Where is it still most likely to fail?
- Which reserve-shelf idea is the best candidate if the active path underdelivers?

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

I do **not** want:
- fluff,
- generic “future work,”
- shallow novelty for novelty’s sake,
- or re-arguing already-settled choices unless you have a genuinely strong reason.

Your goal is to help make Hydra not just more complete, but more formidable and more coherent **after** the active-path / reserve-shelf split has already been made.
