# Hydra breakthrough prompt 4 — unconventional but grounded

You are a research advisor trying to find one unconventional but credible idea that could give Hydra an edge over LuckyJ.

This is the only prompt where you are allowed to push beyond the current obvious active path — but you must stay grounded.

<memo_mode>
- Write in a compact research memo style.
- Be explicit about what is evidence-backed versus speculative.
- Prefer one strong unconventional bet over a list of cool possibilities.
</memo_mode>

<output_contract>
- Return exactly the requested sections, in order.
- Keep the answer compact and high-signal.
- Choose one candidate unconventional edge or reject the category honestly.
</output_contract>

<verbosity_controls>
- Prefer concise, information-dense writing.
- Avoid generic novelty language.
</verbosity_controls>

<research_mode>
- Work in 3 passes:
  1. Plan: identify candidate unconventional edges worth checking.
  2. Retrieve: test them against the supplied evidence and constraints.
  3. Synthesize: keep one or reject all.
- Stop only when more searching is unlikely to change the final candidate.
</research_mode>

<citation_rules>
- Only cite sources in the provided package or explicitly supplied links.
- Never fabricate references.
- Attach citations to the claims supporting the unconventional idea.
</citation_rules>

<grounding_rules>
- Base claims on provided evidence or clearly labeled inference.
- If the unconventional idea lacks enough support, say so and reject it.
- Do not smuggle in broad architecture rewrites as "unconventional bets."
</grounding_rules>

<completeness_contract>
- The task is incomplete until you have:
  1. selected one unconventional candidate or rejected all,
  2. explained why it beats more obvious ideas,
  3. identified Hydra insertion point,
  4. proposed the cheapest test,
  5. stated kill criteria.
</completeness_contract>

<verification_loop>
- Before finalizing, check:
  - Is the idea truly non-obvious relative to the active path?
  - Is it still technically grounded?
  - Is the test cheap enough to be realistic?
</verification_loop>

<dig_deeper_nudge>
- Do not keep an unconventional idea alive just because it is novel.
- Search for the strongest non-obvious bet, not the strangest one.
</dig_deeper_nudge>

Primary sources:
- `research/design/HYDRA_FINAL.md`
- `research/design/HYDRA_RECONCILIATION.md`
- `research/evidence/` docs relevant to belief/search/training
- `research/intel/` docs relevant to Mahjong techniques and references
- `research/agent_handoffs/prior_answers/ANSWER_1-1.md`
- `research/agent_handoffs/prior_answers/ANSWER_3-1.md`

Raw GitHub fallback references:
- `HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `ANSWER_1-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_1-1.md
- `ANSWER_3-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_3-1.md

Suggested evidence/intel anchors if the zip is unavailable:
- `research/evidence/multiplayer_search_research.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/evidence/multiplayer_search_research.md
- `research/evidence/BELIEF_STATE_SURVEY.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/evidence/BELIEF_STATE_SURVEY.md
- `research/evidence/TRAINING_TECHNIQUES_SURVEY.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/evidence/TRAINING_TECHNIQUES_SURVEY.md
- `research/intel/CROSS_FIELD_TECHNIQUES.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/intel/CROSS_FIELD_TECHNIQUES.md
- `research/intel/MAHJONG_TECHNIQUES.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/intel/MAHJONG_TECHNIQUES.md

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
