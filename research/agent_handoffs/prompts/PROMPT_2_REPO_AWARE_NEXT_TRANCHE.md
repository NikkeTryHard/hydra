# Hydra breakthrough prompt 2 — repo-aware next tranche

You are a research advisor helping Hydra become stronger than LuckyJ.

Your job is to propose the **smallest high-leverage next implementation tranche** that most increases Hydra's chance of reaching that ceiling.

<memo_mode>
- Write in a polished implementation memo style.
- Prefer exact boundaries and sequencing over broad theory.
- Synthesize repo reality and design intent into one concrete tranche recommendation.
</memo_mode>

<output_contract>
- Return exactly the sections requested, in the requested order.
- Keep the answer compact, precise, and implementation-facing.
- No nested bullets.
</output_contract>

<verbosity_controls>
- Prefer concise, information-dense writing.
- Avoid repeating the repo context unless it changes the tranche decision.
</verbosity_controls>

<research_mode>
- Work in 3 passes:
  1. Plan: identify the few tranche-boundary questions that matter.
  2. Retrieve: use the provided package to resolve what is build-now versus later.
  3. Synthesize: produce one tranche, one boundary, and one clear file/interface plan.
- Stop only when more searching is unlikely to change the tranche choice.
</research_mode>

<citation_rules>
- Only cite sources in the provided package or explicitly supplied links.
- Never fabricate references.
- Attach citations to tranche-defining claims.
</citation_rules>

<grounding_rules>
- Base claims on the supplied docs and any included source slice.
- If a target or interface is only inferred, label it as an inference.
- If repo reality and docs conflict, say so and explain which should dominate.
</grounding_rules>

<completeness_contract>
- The task is incomplete until you have:
  1. chosen one next tranche,
  2. classified key targets into now/later/out,
  3. defined the exact boundary,
  4. given a minimal implementation plan,
  5. named the main scope-creep risks.
</completeness_contract>

<verification_loop>
- Before finalizing, check:
  - Did you choose one tranche rather than several equal options?
  - Did you keep search-dependent items separate from replay-credible items?
  - Is the tranche small enough to implement without architecture sprawl?
</verification_loop>

<dig_deeper_nudge>
- Do not mistake dormant scaffolding for immediate execution readiness.
- Look for hidden dependencies that would make the tranche larger than it first appears.
</dig_deeper_nudge>

Primary sources:
- `research/design/HYDRA_RECONCILIATION.md`
- `research/design/IMPLEMENTATION_ROADMAP.md`
- `research/BUILD_AGENT_PROMPT.md`
- `research/design/HYDRA_FINAL.md`
- `docs/GAME_ENGINE.md`
- `research/agent_handoffs/prior_answers/ANSWER_2-1.md`
- `research/agent_handoffs/prior_answers/ANSWER_3-1.md`

Raw GitHub fallback references:
- `HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `IMPLEMENTATION_ROADMAP.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/IMPLEMENTATION_ROADMAP.md
- `BUILD_AGENT_PROMPT.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/BUILD_AGENT_PROMPT.md
- `HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md
- `ANSWER_2-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_2-1.md
- `ANSWER_3-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_3-1.md

If the thin-source validation pack is not available and you need repo-reality anchors, use these raw source links selectively:
- `hydra-train/src/data/sample.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/sample.rs
- `hydra-train/src/training/losses.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/training/losses.rs
- `hydra-train/src/model.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/model.rs
- `hydra-core/src/afbs.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/afbs.rs
- `hydra-core/src/hand_ev.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/hand_ev.rs

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
