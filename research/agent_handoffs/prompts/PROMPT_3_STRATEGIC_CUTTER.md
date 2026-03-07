# Hydra breakthrough prompt 3 — strategic cutter

You are a strategic research advisor for Hydra.

Your task is to make Hydra's path to beating LuckyJ **smaller, sharper, and more coherent**.

Primary sources:
- `research/design/HYDRA_FINAL.md`
- `research/design/HYDRA_RECONCILIATION.md`
- `research/design/IMPLEMENTATION_ROADMAP.md`
- `research/design/OPPONENT_MODELING.md`
- `research/design/TESTING.md`
- `research/infrastructure/INFRASTRUCTURE.md`
- `research/agent_handoffs/prior_answers/ANSWER_3-1.md`
- `research/agent_handoffs/prior_answers/ANSWER_1-1.md`
- `research/agent_handoffs/prior_answers/ANSWER_2-1.md`

Treat the reconciled active-path decisions as fixed unless one is clearly catastrophic.

Your job:
1. Audit the active path for the biggest remaining ways it could still fail.
2. Re-rank the reserve shelf harder.
3. Keep at most 3 surviving breakthrough bets.
4. Force final classifications for major ideas:
   - do now
   - do after active path stabilizes
   - reserve only
   - drop

Constraints:
- Do not brainstorm broadly.
- Do not preserve ideas just because they are elegant.
- Do not create a second roadmap from the reserve shelf.
- Be harsher than the current reconciliation memo.

Output format:
1. Active path failure audit
2. Reserve shelf triage
3. Top 3 surviving breakthrough bets
4. Final hard-decision table
5. Final narrowed Hydra recommendation

Success means Hydra ends up easier to execute and harder to misunderstand.
