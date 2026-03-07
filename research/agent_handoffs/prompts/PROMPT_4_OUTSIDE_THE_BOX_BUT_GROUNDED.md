# Hydra breakthrough prompt 4 — unconventional but grounded

You are a research advisor trying to find one unconventional but credible idea that could give Hydra an edge over LuckyJ.

This is the only prompt where you are allowed to push beyond the current obvious active path — but you must stay grounded.

Primary sources:
- `research/design/HYDRA_FINAL.md`
- `research/design/HYDRA_RECONCILIATION.md`
- `research/evidence/` docs relevant to belief/search/training
- `research/intel/` docs relevant to Mahjong techniques and references
- `research/agent_handoffs/prior_answers/ANSWER_1-1.md`
- `research/agent_handoffs/prior_answers/ANSWER_3-1.md`

Your task:
1. Find the single best unconventional idea that is still plausible for Hydra.
2. The idea must satisfy all of:
   - not already the obvious active-path item
   - plausible strength upside against LuckyJ-level systems
   - technically testable without rewriting the whole project
   - grounded in at least adjacent evidence, not pure fantasy
3. Explain exactly how it would fit into Hydra.
4. Give the cheapest experiment to test it.
5. Say clearly why it is still worth trying despite the active-path pruning.

Constraints:
- No broad architecture reboot.
- No AGPL/code-copy ideas.
- No giant compute fantasies without a realistic test path.
- If the idea is too speculative, say so and reject it yourself.

Output format:
1. Candidate unconventional edge
2. Why it might beat more obvious ideas
3. Evidence level
4. Hydra insertion point
5. Cheapest test
6. Kill criteria

Success means you surface one unusually strong non-obvious bet, or conclude honestly that no such bet currently beats the narrowed active path.
