# Hydra breakthrough prompt 2 — repo-aware next tranche

You are a research advisor helping Hydra become stronger than LuckyJ.

Your job is to propose the **smallest high-leverage next implementation tranche** that most increases Hydra's chance of reaching that ceiling.

Primary sources:
- `research/design/HYDRA_RECONCILIATION.md`
- `research/design/IMPLEMENTATION_ROADMAP.md`
- `research/BUILD_AGENT_PROMPT.md`
- `research/design/HYDRA_FINAL.md`
- `docs/GAME_ENGINE.md`
- `research/agent_handoffs/prior_answers/ANSWER_2-1.md`
- `research/agent_handoffs/prior_answers/ANSWER_3-1.md`

If source files are included in the package, use them only to ground repo reality, not to redesign everything.

Assume these are fixed:
- the active path is supervision-first
- Hand-EV is next after supervision closure
- AFBS stays specialist
- broad search expansion is not the next move

Your job:
1. Pick the one next tranche Hydra should build.
2. Separate everything into:
   - build now
   - later after teacher/search infra exists
   - not part of this tranche
3. Define the exact boundary:
   - what data enters
   - what targets are produced
   - what files/interfaces must change
   - what is explicitly out of scope
4. Explain why this tranche is the best move if the goal is eventually beating LuckyJ.

Constraints:
- Do not widen the tranche into a full architecture program.
- Do not activate every dormant head just because it exists.
- Do not assume search-dependent labels already exist.
- Be ruthless about scope control.

Output format:
1. Best single next tranche
2. Build-now vs later table
3. Exact tranche boundary
4. Minimal file/interface plan
5. Main scope-creep risks
6. Why this tranche matters for LuckyJ-level trajectory

Success means a coding agent could implement the tranche without drifting into a giant side quest.
