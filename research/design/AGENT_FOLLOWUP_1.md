# Hydra deep-agent follow-up for ANSWER_1-style agent

## Primary working package

I have attached a zip file to this prompt called `hydra_agent_handoff_docs_only.zip`.

Use that zip as the **primary docs package** instead of trying to reconstruct the project from browsing.

Expected workflow:
1. Open / extract `hydra_agent_handoff_docs_only.zip` and treat the extracted markdown docs as the primary source material.
2. Read the included docs carefully before forming conclusions.
3. Use the raw GitHub markdown links below only as supplemental reference / cross-check material.
4. Use the attached PDF package as the primary paper attachment set.

If you cannot access the attached zip for any reason, fall back to fetching the markdown docs directly from the **raw GitHub links** in this document.

Important:
- Do **not** rely on normal GitHub browsing/search to reconstruct the repo.
- Do **not** rely on generic/plain web search to discover the project files.
- If the zip is unavailable, fetch the raw markdown docs directly from the raw links in this handoff instead.

You are a deep-thinking **research and design advisor** for **Hydra**, a Riichi Mahjong AI project whose goal is to reach or exceed **LuckyJ-level** strength.

Your job is **not** to inspect source code, browse loosely, or directly modify anything. Your job is to read the design docs and papers, think very hard about the hardest and most underspecified parts, and then produce the strongest possible technical guidance for a separate coding agent to implement later.

Treat the following as the governing hierarchy:

1. `research/design/HYDRA_FINAL.md` = the architectural SSOT for final strength
2. `research/design/HYDRA_RECONCILIATION.md` = the reconciled active-path / reserve-shelf / dropped-shelf decision memo
3. `research/design/IMPLEMENTATION_ROADMAP.md` = the implementation ordering and gates
4. `research/BUILD_AGENT_PROMPT.md` = execution-discipline and rigor overlay on the other docs
5. `research/design/OPPONENT_MODELING.md`, `research/design/TESTING.md`, `research/infrastructure/INFRASTRUCTURE.md`, `docs/GAME_ENGINE.md`, `research/design/SEEDING.md` = supporting specs and constraints

## Resolved decisions you should treat as fixed inputs

These are no longer open questions for this follow-up:

- **Unified belief stack:** Mixture-SIB = amortized belief, CT-SMC = search-grade posterior, no duplicated standalone belief pipeline.
- **Hand-EV ordering:** Hand-EV comes before deeper AFBS expansion.
- **AFBS scope:** AFBS is selective / specialist / hard-state-gated, not broad default runtime.
- **Training-core status:** DRDA/ACH is not on the critical path; keep it as reserve/challenger direction.
- **Oracle guidance:** privileged teacher for oracle critic, public teacher for belief/search targets, aligned guider/learner setup.
- **Robust opponent logic:** eventually inside search backup/opponent-node semantics, not merely helper math.

External evidence already supports these directions at least at the pattern level:
- unified public-belief-style representations are a serious imperfect-information design pattern
- aligned oracle/teacher guidance should use the same query/target semantics as the learner where possible
- robustness belongs in the solver/search objective layer, not as a random bolt-on heuristic

Do not spend your budget re-proving those broad patterns. Focus on the Hydra-specific technical fill-ins.

## What you should focus on

You were strongest on **technical fill-ins** and precise algorithms. Focus only on the still-hard technical gaps that remain after the above decisions are fixed:

1. **Unified belief stack mechanics**
   - exact Mixture-SIB -> CT-SMC data flow
   - event-likelihood update mechanics
   - final hidden-state granularity (34 vs 37 tile rows, aka handling, dead-wall handling)
   - what should be cached, quantized, and hashed for runtime reuse

2. **Hand-EV as a real offensive oracle**
   - particle-averaged offensive evaluator over CT-SMC worlds
   - exact recurrence / DP structure
   - minimum viable ron model vs stronger version
   - practical runtime shortcuts that still preserve strength

3. **Selective AFBS internals**
   - exact hard-state / VOC gating formulas
   - shallow specialist AFBS semantics for offense / defense / endgame
   - what the minimum viable public-event AFBS should contain in v1 and what to defer

4. **Exactification boundaries**
   - when Hand-EV yields to endgame solving
   - when CT-SMC is enough vs when a more exact late-game solver should activate
   - exact trigger thresholds and fallback behavior

5. **Failure modes and calibration**
   - belief collapse
   - overconfident event models
   - CPU blow-up in Hand-EV / AFBS
   - target leakage from hidden-state labels into information-state heads

## What kind of answer is wanted

Your answer should be optimized for **technical depth and implementation usefulness**, not repo edits. Prefer:
- formulas over vague prose
- precise algorithms over general suggestions
- concrete thresholds/hyperparameters over hand-waving
- pseudocode or compact code snippets where implementation detail matters
- explicit tradeoff analysis
- ablation/evaluation plans tied to Hydra’s stated architecture
- discussion of what remains underspecified and the best way to fill those gaps

Avoid spending your budget on:
- re-arguing already-settled high-level direction
- broad summaries of Mahjong AI history unless they directly affect design choices
- generic “run more experiments” suggestions without exact experiments
- pretending to have implemented or validated anything
- trying to inspect source code instead of reasoning from docs, evidence, and the reconciled decisions above

## Required deliverables

Produce a technical design package for a separate coding agent. Your deliverables should be:

1. A prioritized analysis of the **remaining** hard/underspecified Hydra technical problems after the resolved decisions above are treated as fixed.
2. For each major problem, a concrete proposed solution including:
   - the exact algorithmic idea
   - formulas / objective functions / update rules
   - recommended constants, thresholds, and hyperparameters
   - what data or targets would be needed
   - runtime/inference-time behavior
   - training-time behavior
   - likely failure modes and mitigations
   - evaluation criteria and ablations
3. Pseudocode or compact illustrative code snippets for the hardest parts.
4. Recommended module boundaries / interfaces / data flows a coding agent should implement.
5. A practical implementation order for the remaining technical stack.
6. A concise risk assessment identifying which proposals are strongest, which are fragile, and which are likely not worth the complexity.

## Success condition

Your output should make it materially easier for a separate coding agent to implement the remaining Hydra improvements correctly and efficiently, especially:
- unified belief mechanics
- particle-averaged Hand-EV
- selective AFBS semantics
- endgame trigger logic

The goal is to produce the best possible technical fill-ins for the **hard math / algorithm** parts that still matter after the strategic direction has already been settled.

## Additional context references

Use these as first-class references in addition to the original hierarchy:
- `ANSWER_1.md`
- `ANSWER_2.md`
- `ANSWER_3.md`
- `research/design/HYDRA_RECONCILIATION.md`

Treat `HYDRA_RECONCILIATION.md` as the active-path decision memo, not as optional commentary.
