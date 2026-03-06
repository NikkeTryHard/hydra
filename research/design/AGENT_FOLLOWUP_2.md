# Hydra deep-agent follow-up for ANSWER_2-style agent

## Primary working package

I have attached a zip file to this prompt called `hydra_agent_handoff_source_only.zip`.

Use that zip as the **primary codebase snapshot** instead of trying to discover or clone the repository first.

Expected workflow:
1. Open / extract `hydra_agent_handoff_source_only.zip` and treat the extracted contents as the working project.
2. Read the included markdown docs from the zip first.
3. Use the raw GitHub links below only as supplemental reference / cross-check material.
4. Use the attached PDF package as the primary paper attachment set.

If you cannot access the attached zip for any reason, fall back to fetching the repository files directly from the **raw GitHub file links** in this document.

Important:
- Do **not** rely on normal GitHub browsing/search to reconstruct the repo.
- Do **not** rely on generic/plain web search to discover the project files.
- If the zip is unavailable, fetch the raw files directly from the raw GitHub links in this handoff instead.

You are a deep-thinking **research and design advisor** working on **Hydra**, a Rust-first Riichi Mahjong AI whose goal is to reach or exceed **LuckyJ-level** strength.

Your job is **not** to browse loosely or brainstorm from scratch, and it is also **not** to directly integrate changes into the repository. Your job is to think very hard about the remaining unsolved high-leverage problems, then produce the strongest possible technical guidance for a separate coding agent to implement.

Treat the following as the governing hierarchy:

1. `research/design/HYDRA_FINAL.md` = the architectural SSOT for final strength
2. `research/design/HYDRA_RECONCILIATION.md` = the active-path / reserve-shelf / dropped-shelf decision memo
3. `research/design/IMPLEMENTATION_ROADMAP.md` = step-by-step implementation and gates
4. `research/BUILD_AGENT_PROMPT.md` = execution discipline overlay on all docs
5. `research/design/OPPONENT_MODELING.md`, `research/design/TESTING.md`, `research/infrastructure/INFRASTRUCTURE.md`, `docs/GAME_ENGINE.md`, `research/design/SEEDING.md` = supporting specs and constraints

## Current repo reality you must account for

Hydra already has many advanced modules by name and partial implementation:
- fixed-superset 192-channel encoder with Group C / D presence masks
- CT-SMC exact DP sampler
- Mixture-SIB / Sinkhorn support code
- AFBS tree scaffolding
- Hand-EV module
- endgame module
- robust opponent math utilities
- train-side model/head/loss scaffolding

But the main remaining blocker is still **integration and realism**, not mere file absence.

## Resolved decisions you should treat as fixed inputs

These are no longer open questions for this follow-up:

- **Unified belief stack:** Mixture-SIB + CT-SMC, no duplicate standalone belief path.
- **Hand-EV timing:** Hand-EV realism comes before deeper AFBS expansion.
- **AFBS scope:** selective / specialist / hard-state-gated, not broad mainline search.
- **Training-core status:** DRDA/ACH is not the mainline foundation; keep as reserve/challenger branch.
- **Oracle guidance:** privileged oracle teacher only for oracle-critic-like targets; public teacher for belief/search targets; aligned guider/learner setup.

External evidence already supports these broad patterns:
- public-belief-style representations are a valid main substrate for learning/search in imperfect-information systems
- aligned privileged guidance should stay close to the learner's query/target semantics
- robustness should live in search/solver logic rather than only as shallow post-hoc heuristics

Do not spend your budget re-litigating those high-level patterns. Focus on Hydra-specific implementation closure.

## Highest-priority gaps you must analyze deeply

You were strongest on **repo-aware loop closure**. Focus on turning the resolved direction above into a concrete implementation blueprint for the current codebase.

1. **Advanced supervision loop closure**
   - specify the exact data path needed to make advanced targets real end-to-end
   - identify which advanced targets are replay-credible now vs which require teacher search/belief generation
   - define staged activation order for:
     - oracle critic
     - belief fields / mixture weights
     - opponent hand type
     - `ΔQ`
     - safety residual
     - ExIt targets

2. **Canonical data/target boundaries**
   - define the exact boundary between replay-derived labels, bridge-derived labels, teacher-generated labels, and runtime-only features
   - specify what should flow through `MjaiSample`, batch collation, `HydraTargets`, and any new helper structs
   - make presence/absence semantics explicit for optional advanced targets

3. **Public-teacher vs privileged-teacher pipeline**
   - define exactly which targets are privileged-only and which must be information-state/public-teacher targets
   - give a concrete teacher-generation workflow that a coding agent could implement in phases

4. **AFBS loop-closing as an implementation problem**
   - not broad redesign-from-scratch
   - instead: what exact interfaces, caches, labels, and leaf outputs are needed so AFBS becomes useful to training and inference in stages

5. **Hand-EV / endgame / robust-opponent rollout order**
   - given the current codebase state, what exact tranche ordering would close loops fastest without fake progress?

## Additional constraints

- **Do not copy or derive code from `Mortal-Policy/`** or other AGPL sources.
- Reference-only is fine; code derivation is not.
- Maintain Hydra’s Rust conventions, zero-warning policy, and library-code safety rules.
- Preserve engine performance. Do not casually add hot-path regressions.
- Respect the reconciled architecture unless the docs clearly require a correction.

## What kind of answer is wanted

Your answer should be optimized for **technical depth and implementation usefulness**, not repo edits. Prefer:
- formulas where target definitions matter
- precise dataflow / interface guidance
- concrete thresholds/hyperparameters over hand-waving
- pseudocode / compact code snippets where edge cases matter
- explicit tradeoff analysis
- ablation/evaluation plans tied to Hydra’s actual architecture

Avoid spending your budget on:
- re-litigating already-resolved architecture choices
- generic motivational advice
- broad summaries of known Mahjong AI history
- pretending to have implemented or validated code changes

Assume that a separate coding agent will use your response as the implementation blueprint.

## Required deliverables

Produce a technical design package for a separate coding agent. Your deliverables should be:

1. A prioritized analysis of the remaining highest-leverage integration/realism weaknesses in Hydra.
2. For each major gap, a concrete proposed solution including:
   - exact data/target requirements
   - exact interface boundaries
   - runtime vs training-time behavior
   - staged rollout order
   - evaluation criteria and ablations
3. Pseudocode or compact code snippets for the hardest integration points.
4. Recommended interfaces between modules where integration is unclear.
5. A practical implementation order for a coding agent to follow now.
6. A concise risk assessment explaining what is most likely to fail, overfit, be too slow, or be too weak.

## Success condition

Your output should make it materially easier for a separate coding agent to implement the next Hydra tranches correctly and efficiently. The goal is to produce the best possible **repo-aware implementation blueprint** for closing the loops that are still only half-alive in code.

## Additional context references

Use these as first-class references in addition to the original hierarchy:
- `ANSWER_1.md`
- `ANSWER_2.md`
- `ANSWER_3.md`
- `research/design/HYDRA_RECONCILIATION.md`

Treat `HYDRA_RECONCILIATION.md` as the active-path decision memo and assume the codebase has already been reconciled against the most dangerous doc drift.
