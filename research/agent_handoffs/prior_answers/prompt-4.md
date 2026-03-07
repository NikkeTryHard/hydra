# Hydra breakthrough prompt 4 — unconventional but grounded

Attach these files to the model:
- `hydra_breakthrough_docs_pack.zip`
- `deep_agent_20_pdfs.zip`

Zip structure the model should expect in `hydra_breakthrough_docs_pack.zip`:
- core design docs under `research/design/`
- runtime summary under `docs/`
- infra summary under `research/infrastructure/`
- prior answer archive under `research/agent_handoffs/prior_answers/`
- prompt files under `research/agent_handoffs/prompts/`

Zip structure the model should expect in `deep_agent_20_pdfs.zip`:
- paper PDFs under `deep_agent_20_pdfs/`

The attached zips should be sufficient by themselves. Use raw links only if the attachments are inaccessible or corrupted.
For this task, do not go looking for the thin-source validation pack unless the docs pack is missing and you have a specific reason to believe source validation is necessary.

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


1. Candidate unconventional edge

Confidence-gated safe exploitation: keep Hydra’s default policy/search robust, but add a small root-level exploit residual over the existing opponent-archetype bank when the opponent-style posterior is sharp, belief quality is healthy, and the state is still on-support. This is not the active mainline; it is a reserve/challenger idea. The grounding is that Hydra already has an OLSS-style archetype set with posterior weights, a league with anchors/exploiters, and a concrete gated residual proposal; the speculative step is that this becomes a separator against LuckyJ-like populations rather than just another robustness layer. (research/design/HYDRA_RECONCILIATION.md:214-228; research/design/HYDRA_FINAL.md:232-239, 243-247, 326-327; research/agent_handoffs/prior_answers/ANSWER_3-1.md:345-353, 547-574; research/design/IMPLEMENTATION_ROADMAP.md:685-689, 804-805)

2. Why it might beat more obvious ideas

It does not beat the obvious active-path items as the next coding tranche; target/loss closure and Hand-EV should still come first. It might beat the more obvious reserve moves as a separator because endgame exactification is powerful but narrow, and fuller robust-opponent/AFBS work mostly deepens machinery Hydra already shares with LuckyJ’s OLSS-style direction; safe exploitation is one of the few surviving bets that can create asymmetric upside across many hands by turning style deviations into EV rather than only reducing error. The reconciliation memo’s reserve-shelf language matters here: this class of idea is deferred because of sequencing complexity, not because it is unsound. (research/design/HYDRA_RECONCILIATION.md:175-205, 214-228; research/agent_handoffs/prior_answers/ANSWER_3-1.md:239-261, 345-353; research/design/HYDRA_FINAL.md:239)

3. Evidence level

Medium. Evidence-backed: Hydra’s own design already treats multiplayer Mahjong as exploitable, already allocates an archetype-weighted opponent model, and prior handoffs give a concrete safe–exploit mixing rule and hard gating thresholds. Adjacent evidence: OLSS shows that limiting opponent strategy can materially improve Mahjong online performance, and learned confidence/OOD methods show that cheap confidence signals can be trained and calibrated without bespoke OOD labels. Speculative: none of the supplied sources proves that this exact mechanism wins in 4-player Riichi, so the whole bet lives or dies on posterior calibration and held-out-mixture robustness, not on novelty. (research/design/HYDRA_FINAL.md:232-239, 376; research/agent_handoffs/prior_answers/ANSWER_3.md:795-816, 988-992; deep_agent_20_pdfs/olss_2023.pdf:12-32; deep_agent_20_pdfs/ood_confidence_2018.pdf:5-15, 22-29)

4. Hydra insertion point

Insert it at the existing opponent-archetype/SaF boundary, not as a new search stack. Concretely: use the current archetype-conditioned values (Q_i(a)), form an exploitation advantage relative to the balanced archetype, and add only a bounded root-level logit residual when the gate passes; the prior handoff’s cheap gate is already concrete: sharp posterior, decent ESS, low posterior entropy. This fits directly where Hydra already has `num_archetypes=4`, `tau_arch`, and a root-level residual path, and it can be trained/evaluated against the existing anchor/exploiter league rather than a new architecture. (research/design/HYDRA_FINAL.md:243-247; research/design/IMPLEMENTATION_ROADMAP.md:685-689, 804-805; research/agent_handoffs/prior_answers/ANSWER_3-1.md:547-574)

5. Cheapest test

Do the smallest challenger, not a new system. Reuse four existing league policies as a style bank, train only a lightweight public-history style posterior plus confidence score, and add the root-level exploit residual with the hard gate; for the first pass, approximate archetype-conditional values with shallow/cached root evaluation rather than full robust-opponent search. Then run duplicate bundles against style-specific anchor pools, held-out mixed pools, and balanced anchors; success is gain on style-biased pools without measurable collapse on balanced or held-out mixtures. For the confidence/OOD side, use misclassified in-distribution histories as proxy negatives rather than building a separate OOD dataset. (research/agent_handoffs/prior_answers/ANSWER_3-1.md:353, 547-574; research/design/IMPLEMENTATION_ROADMAP.md:804-805; deep_agent_20_pdfs/ood_confidence_2018.pdf:9-15)

6. Kill criteria

Kill it if the style posterior is not actually trustworthy, or if the gains are fake. Concretely: stop if style-posterior calibration stays poor, if OOD gating does not separate held-out opponent mixtures, if duplicate gains appear only against one seen anchor family but collapse against balanced/held-out mixtures, or if the module adds more than about 10% latency or 20% training cost for less than about 0.5 rank-point/game. If exploit mode only works with aggressive mixing that visibly increases 4th-place collapse, reject the category and leave it on the shelf. (research/agent_handoffs/prior_answers/ANSWER_3.md:978-992, 998-1012; research/agent_handoffs/prior_answers/ANSWER_3-1.md:681-690)
