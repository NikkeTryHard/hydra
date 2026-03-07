# Hydra breakthrough prompt 1 — technical ceiling

You are a research advisor for Hydra, a Riichi Mahjong AI project trying to become stronger than LuckyJ.

Your task is to identify the **single strongest technical breakthrough path** still available inside Hydra's reconciled architecture.

Use these as primary sources:
- `research/design/HYDRA_FINAL.md`
- `research/design/HYDRA_RECONCILIATION.md`
- `research/design/IMPLEMENTATION_ROADMAP.md`
- `research/design/OPPONENT_MODELING.md`
- `docs/GAME_ENGINE.md`
- `research/design/SEEDING.md`
- `research/design/TESTING.md`
- prior answer files in `research/agent_handoffs/prior_answers/`

Assume these are already settled:
- unified belief stack = Mixture-SIB + CT-SMC
- Hand-EV before deeper AFBS
- AFBS is selective/specialist, not broad default runtime
- DRDA/ACH is off the critical path
- supervision-first before search-expansion-first

Your job:
1. Identify the highest-upside remaining breakthrough that is still compatible with the reconciled active path.
2. Explain why it could matter specifically in Mahjong.
3. Give the exact technical shape:
   - formulas
   - update rules
   - thresholds
   - approximations
   - failure modes
4. Give the cheapest meaningful experiment that would falsify or support it.
5. Compare it against the best obvious alternative and say why your chosen path wins.

Constraints:
- Do not inspect code unless a raw source file is explicitly included in the package.
- Do not propose broad architecture sprawl.
- Do not give a menu of many equal ideas.
- Pick one main breakthrough path and defend it.

Output format:
1. Executive claim
2. Why this is the best remaining breakthrough
3. Technical specification
4. Failure modes
5. Cheapest decisive test
6. Why the main alternatives lose

Success means your answer is concrete enough that a coding agent could start designing toward it immediately.
