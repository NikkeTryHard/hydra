# Hydra breakthrough prompt 3 — strategic cutter

Attach this zip to the model:
- `hydra_breakthrough_docs_pack.zip`

Zip structure the model should expect:
- core design docs under `research/design/`
- runtime summary under `docs/`
- infra summary under `research/infrastructure/`
- prior answer archive under `research/agent_handoffs/prior_answers/`
- prompt files under `research/agent_handoffs/prompts/`

The zip should be sufficient by itself. Use raw links only if the attachment is inaccessible or corrupted.
For this task, do not go looking for the thin-source validation pack unless the docs pack is missing and you have a specific reason to believe source validation is necessary.

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

Raw GitHub fallback references:
- `HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `IMPLEMENTATION_ROADMAP.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/IMPLEMENTATION_ROADMAP.md
- `OPPONENT_MODELING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md
- `TESTING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md
- `INFRASTRUCTURE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/infrastructure/INFRASTRUCTURE.md
- `ANSWER_1-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_1-1.md
- `ANSWER_2-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_2-1.md
- `ANSWER_3-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_3-1.md

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


## 1. Active path failure audit

1. **Catastrophic if not cut: authority drift.** `HYDRA_RECONCILIATION.md` says the first move is narrow target/loss closure and explicitly defers broad AFBS expansion, but the build prompt still orders a line-by-line 12-step full-stack build; the roadmap still specifies from-scratch AFBS/robust-opponent/self-play construction; infrastructure still operationalizes PPO-era 40-block assumptions; testing still smokes `[1,85,34]` tensors. A coding agent can therefore “follow the docs” and still build the wrong Hydra. **Cut:** make `HYDRA_RECONCILIATION.md` the temporary sequencing authority and demote the build prompt, roadmap, infra, and testing docs to reference status until rewritten. (HYDRA_RECONCILIATION.md:90-101, 329-355; BUILD_AGENT_PROMPT.md:5-16, 47-92; IMPLEMENTATION_ROADMAP.md:1-16, 662-790; INFRASTRUCTURE.md:517-560, 578-613, 798-810; TESTING.md:226-246)

2. **Most likely stall: tranche-1 scope leak.** Repo reality is not “missing advanced components”; it is “advanced surfaces exist, but losses are zero and batches are mostly baseline-only.” Reconciliation still leaves room to pull belief fields, mixture weights, and opponent-hand-type labels into the same tranche “if credible.” That is too loose. **Cut:** tranche 1 is exactly `ExIt`, `ΔQ`, and `safety_residual`, with explicit provenance/presence. No oracle-path work, no belief teacher, no Hand-EV rewrite, no AFBS semantics rewrite in that tranche. (HYDRA_RECONCILIATION.md:58-88, 337-383, 400-423; ANSWER_2-1.md:159-169, 171-188, 192-295)

3. **Most important technical blank: belief supervision can still be implemented wrong.** `HYDRA_FINAL.md` exposes Mixture-SIB fields and mixture logits as model outputs, but raw Sinkhorn external fields are not a stable supervision object; they are gauge-like and non-identifiable. **Cut:** if belief supervision survives, supervise projected beliefs/gauge-fixed marginals only. Raw-field regression is dropped now, not debated later. (HYDRA_FINAL.md:115-117, 121-157; ANSWER_3-1.md:225-226, 305-313)

4. **Likely underdelivery path: Hand-EV stays “better heuristics” instead of becoming a real oracle.** Reconciliation correctly moves Hand-EV ahead of deeper AFBS, and the current repo already threads it into encoder paths, but the existing implementation is still heuristic. **Cut:** do not call Hand-EV done until it is particle-averaged over CT-SMC worlds, uses bounded offensive DP, includes a simplified ron model, and clears both offensive-value and latency gates. (HYDRA_RECONCILIATION.md:121-132, 260-267; ANSWER_1-1.md:330-423, 462-470)

5. **Search can still teach noise.** The settled posture is “AFBS specialist, not default runtime,” but that only matters if search labels are filtered hard. **Cut:** no search-derived supervision outside a deterministic hard-state gate plus trust filters on visits, expanded mass, ESS, and variance. Broad public-belief search stays dropped. (HYDRA_RECONCILIATION.md:161-171, 230-240, 269-275; ANSWER_1-1.md:476-498, 604-623; ANSWER_3-1.md:231-232)

## 2. Reserve shelf triage

1. **Stronger endgame exactification — keep highest; do after active path stabilizes.** **Plausible.** It is selective, bounded, and the cleanest high-leverage reserve idea. (HYDRA_FINAL.md:202-216; ANSWER_1-1.md:680-759; ANSWER_3-1.md:239-245)

2. **Structured belief updates — reserve only.** **Plausible contingency.** Belief misspecification is the stated core risk, so this stays alive only as the fallback if the unified Mixture-SIB + CT-SMC stack misses calibration or latency gates. (HYDRA_FINAL.md:374-379; HYDRA_RECONCILIATION.md:211-223; ANSWER_3-1.md:263-269)

3. **Robust-opponent search backups — reserve only.** **Plausible but downstream.** They only matter if selective AFBS already proves positive; otherwise they are sophistication on top of noise. (HYDRA_FINAL.md:220-239; ANSWER_3-1.md:247-253)

4. **Confidence-gated exploitation — reserve only, low priority.** **Speculative-plausible.** Cheap challenger path, but evidence is indirect and it dies with poor opponent-posterior calibration. (HYDRA_RECONCILIATION.md:214-215; ANSWER_3-1.md:255-261)

5. **Narrow AFBS semantic refinements — reserve only.** **Plausible.** Preserve public-event semantics and hard-state expansion rules; do not preserve “deeper AFBS” as an open bucket. (HYDRA_RECONCILIATION.md:218-221; ANSWER_3-1.md:279-285)

6. **Richer latent opponent posterior / extra opponent heads — drop.** **Weak.** The bottleneck is not missing outputs, and proposed label paths such as call-intent are survivorship-biased. (HYDRA_RECONCILIATION.md:293-300, 312-315; ANSWER_3-1.md:271-277; OPPONENT_MODELING.md:457-474)

7. **Optimizer/game-theory additions beyond the existing ACH/DRDA assumption — drop.** **Weak for current sequencing.** Keep the architectural assumption in SSOT, but spend no present planning attention here. (HYDRA_RECONCILIATION.md:277-283; ANSWER_3-1.md:287-293)

8. **Deception reward and similar novelty-heavy ToM modules — drop.** **Speculative.** The proposal itself says the coefficient is arbitrary and lacks prior tuning basis. (OPPONENT_MODELING.md:738-740, 746-769; ANSWER_3-1.md:295-301)

## 3. Top 3 surviving breakthrough bets

Selective AFBS itself does **not** count as a surviving breakthrough bet. It is a controlled delivery mechanism for later search labels, not a fourth initiative.

1. **Hand-EV realism.** **Proven-enough to act.** This survives because it is already wired into Hydra, explicitly prioritized ahead of deeper AFBS, and is the cheapest offensive multiplier on the board. The surviving bet is not “more Hand-EV heuristics”; it is a particle-averaged bounded offensive oracle with ron modeling. (HYDRA_FINAL.md:18-24; HYDRA_RECONCILIATION.md:121-132, 260-267; ANSWER_1-1.md:330-423; ANSWER_3-1.md:315-323)

2. **Projected belief supervision on the current Mixture-SIB + CT-SMC stack.** **Plausible.** This survives because belief misspecification is Hydra’s core stated risk, and this strengthens the existing unified stack instead of spawning another one. The bet is projected beliefs/gauge-fixed marginals, not raw Sinkhorn fields. (HYDRA_FINAL.md:115-117, 374-379; ANSWER_3-1.md:225-226, 305-313)

3. **Stronger endgame exactification.** **Plausible and bounded.** This survives because late-game placement swings are leverage-heavy and the compute envelope is naturally narrow. It is the best post-stabilization ceiling add-on. (HYDRA_FINAL.md:202-216; ANSWER_1-1.md:680-759; ANSWER_3-1.md:239-245, 325-333)

## 4. Final hard-decision table

| Idea                                                                                         | Classification                  | Hard reason                                                                                                                                     |
| -------------------------------------------------------------------------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `HYDRA_RECONCILIATION.md` overrides build prompt, roadmap, infra, and testing for sequencing | do now                          | Fixes the single biggest execution risk: doc-authority drift. (HYDRA_RECONCILIATION.md:90-101, 329-355; BUILD_AGENT_PROMPT.md:5-16)             |
| Narrow advanced-target closure: `ExIt`, `ΔQ`, `safety_residual` only                         | do now                          | Smallest real loop closure; everything else stays absent. (HYDRA_RECONCILIATION.md:337-383; ANSWER_2-1.md:159-169, 268-294)                     |
| Projected belief supervision, not raw field regression                                       | do now                          | Prevents mathematically wrong belief labels from entering the roadmap later. (HYDRA_FINAL.md:115-117; ANSWER_3-1.md:225-226)                    |
| Hand-EV realism / bounded offensive oracle                                                   | do now                          | Cheapest offensive multiplier already wired into the live system. (HYDRA_RECONCILIATION.md:121-132, 260-267)                                    |
| Trust-gated selective AFBS label generation                                                  | do after active path stabilizes | The only acceptable search expansion path; no broad search identity. (HYDRA_RECONCILIATION.md:162-171, 269-275; ANSWER_1-1.md:476-498, 604-623) |
| Stronger endgame exactification                                                              | do after active path stabilizes | Best bounded post-stabilization add-on. (HYDRA_FINAL.md:202-216; ANSWER_3-1.md:239-245)                                                         |
| Structured belief updates contingency                                                        | reserve only                    | Activate only if the unified stack fails belief-quality or latency gates. (HYDRA_FINAL.md:376-379; ANSWER_3-1.md:263-269)                       |
| Robust-opponent search backups                                                               | reserve only                    | Downstream sophistication; not part of the smaller live path. (HYDRA_FINAL.md:220-239; ANSWER_3-1.md:247-253)                                   |
| Confidence-gated exploitation                                                                | reserve only                    | Cheap challenger only; evidence is indirect. (ANSWER_3-1.md:255-261)                                                                            |
| Richer latent opponent posterior / extra opponent heads                                      | drop                            | Current bottleneck is not missing outputs; proposed label path is weak. (HYDRA_RECONCILIATION.md:312-315; OPPONENT_MODELING.md:457-474)         |
| Broad public-belief search / broad AFBS rollout                                              | drop                            | Explicitly rejected as immediate identity: too expensive and too confusing. (HYDRA_RECONCILIATION.md:230-240, 269-275)                          |
| Optimizer/game-theory detours beyond the existing assumption                                 | drop                            | High opportunity cost, low current leverage. (HYDRA_RECONCILIATION.md:277-283; ANSWER_3-1.md:287-293)                                           |
| Deception reward / novelty-heavy ToM extras                                                  | drop                            | Arbitrary tuning, weak basis, pure scope bait. (OPPONENT_MODELING.md:738-740, 746-769; ANSWER_3-1.md:295-301)                                   |

## 5. Final narrowed Hydra recommendation

Hydra does **not** beat LuckyJ by matching LuckyJ’s breadth on a drifting codebase. It beats LuckyJ by making fewer Hydra ideas real: one authority hierarchy, one narrow target-closure tranche, one correct belief-supervision story, and one real offensive oracle. After those gates clear, Hydra gets exactly one bounded ceiling add-on first—endgame exactification—and only then one tightly trust-gated search-teacher path. Robust-opponent backups, exploitation layers, structured-belief contingencies, and all extra-head or optimizer/ToM detours stay shelved. That is the smallest plausible path that attacks Hydra’s actual bottlenecks—partially closed loops, belief correctness, offensive realism, and scope drift—instead of recreating a second overextended roadmap. (HYDRA_RECONCILIATION.md:35-41, 301-315, 329-355; ANSWER_2-1.md:159-169, 283-294; ANSWER_3-1.md:607-635, 703-709)
