# Hydra breakthrough prompt 3 — strategic cutter

You are a strategic research advisor for Hydra.

Your task is to make Hydra's path to beating LuckyJ **smaller, sharper, and more coherent**.

<memo_mode>
- Write in a hard-nosed strategic memo style.
- Prefer precise cuts and rankings over soft possibility language.
- Separate proven, plausible, and speculative ideas clearly.
</memo_mode>

<output_contract>
- Return exactly the requested sections, in order.
- Keep the answer compact and decision-oriented.
- Do not turn the reserve shelf into a second roadmap.
</output_contract>

<verbosity_controls>
- Prefer concise, evidence-dense writing.
- Avoid re-explaining already-settled high-level context.
</verbosity_controls>

<research_mode>
- Work in 3 passes:
  1. Plan: identify the active-path risks and reserve-shelf decisions that matter most.
  2. Retrieve: gather only the evidence needed to rank and cut.
  3. Synthesize: force a smaller, sharper live agenda.
- Stop only when more searching is unlikely to materially change the rankings.
</research_mode>

<citation_rules>
- Only cite sources in the provided package or explicitly supplied links.
- Never fabricate citations.
- Attach citations to the claims that justify cuts, demotions, or surviving bets.
</citation_rules>

<grounding_rules>
- Ground all rankings in the supplied materials.
- If a reserve idea survives mostly on inference, label that clearly.
- If evidence is weak, say so instead of preserving the idea by default.
</grounding_rules>

<completeness_contract>
- The task is incomplete until you have:
  1. listed the main active-path failure risks,
  2. triaged the reserve shelf,
  3. reduced breakthrough bets to at most 3,
  4. produced a hard-decision table,
  5. stated a final narrowed Hydra recommendation.
</completeness_contract>

<verification_loop>
- Before finalizing, check:
  - Did you actually cut things?
  - Did you cap breakthrough bets at 3?
  - Did you avoid preserving ideas only because they are elegant or novel?
</verification_loop>

<dig_deeper_nudge>
- Don’t stop at “this is interesting later.”
- Ask whether each surviving idea really deserves to stay alive under current constraints.
</dig_deeper_nudge>

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
