# Hydra breakthrough prompt 1 — technical ceiling

Attach this zip to the model:
- `hydra_breakthrough_docs_pack.zip`

Zip structure the model should expect:
- core design docs under `research/design/`
- runtime summary under `docs/`
- infra summary under `research/infrastructure/`
- prior answer archive under `research/agent_handoffs/prior_answers/`
- prompt files under `research/agent_handoffs/prompts/`

The zip should be sufficient by itself. Use raw links only if the attachment is inaccessible or corrupted.

You are a research advisor for Hydra, a Riichi Mahjong AI project trying to become stronger than LuckyJ.

Your task is to identify the **single strongest technical breakthrough path** still available inside Hydra's reconciled architecture.

<memo_mode>
- Write in a polished, professional memo style.
- Prefer exact, evidence-backed conclusions over generic hedging.
- Synthesize across sources rather than summarizing each one separately.
</memo_mode>

<output_contract>
- Return exactly the sections requested, in the requested order.
- Keep the answer compact but information-dense.
- Do not repeat the prompt or restate settled assumptions unless they change the conclusion.
</output_contract>

<verbosity_controls>
- Prefer concise, high-density writing.
- Avoid filler, repetition, and generic motivation.
- Do not shorten so aggressively that formulas, evidence, or failure modes disappear.
</verbosity_controls>

<research_mode>
- Work in 3 passes:
  1. Plan: identify the 3-5 technical questions that matter most for the breakthrough candidate.
  2. Retrieve: collect evidence from the provided package and follow 1-2 second-order leads if needed.
  3. Synthesize: choose one main path, resolve contradictions, and produce the final memo.
- Stop only when more searching is unlikely to change the conclusion.
</research_mode>

<citation_rules>
- Only cite sources available in the provided package or explicitly supplied links.
- Never fabricate citations, URLs, or quote spans.
- Attach citations to the specific claims they support.
</citation_rules>

<grounding_rules>
- Base claims only on provided documents, supplied links, or explicit evidence gathered during the task.
- If a claim is an inference rather than directly stated, label it as an inference.
- If sources conflict, state the conflict explicitly and resolve it.
</grounding_rules>

<completeness_contract>
- The task is incomplete until you have:
  1. named one main breakthrough path,
  2. specified its technical mechanism,
  3. identified failure modes,
  4. proposed a decisive cheap test,
  5. explained why the main alternatives lose.
</completeness_contract>

<verification_loop>
- Before finalizing, check:
  - Did you pick one main breakthrough rather than a menu?
  - Are the key claims grounded in the provided materials?
  - Are formulas/thresholds concrete enough to guide implementation?
</verification_loop>

<dig_deeper_nudge>
- Do not stop at the first plausible technical upgrade.
- Look for second-order constraints, hidden costs, and failure-triggering assumptions.
</dig_deeper_nudge>

Use these as primary sources:
- `research/design/HYDRA_FINAL.md`
- `research/design/HYDRA_RECONCILIATION.md`
- `research/design/IMPLEMENTATION_ROADMAP.md`
- `research/design/OPPONENT_MODELING.md`
- `docs/GAME_ENGINE.md`
- `research/design/SEEDING.md`
- `research/design/TESTING.md`
- prior answer files in `research/agent_handoffs/prior_answers/`

Raw GitHub fallback references:
- `HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `IMPLEMENTATION_ROADMAP.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/IMPLEMENTATION_ROADMAP.md
- `OPPONENT_MODELING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md
- `GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md
- `SEEDING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/SEEDING.md
- `TESTING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md
- `ANSWER_1-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_1-1.md
- `ANSWER_3-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_3-1.md

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
