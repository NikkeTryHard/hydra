# Hydra genie follow-up prompt — hidden-world lane pass two

Use this as the follow-up prompt for the same genie after the first hidden-world packet. The first pass already did the diagnosis, candidate-object narrowing, and broad cross-field scan. This pass should **not** spend its budget re-explaining the lane. Its job is to turn that narrowed lane into the strongest actual strength-producing recipe.

You are the genie again. Do not act like a cautious repo maintainer whose job is only to preserve current Hydra assumptions. Your job is to maximize Hydra strength. If the strongest surviving design requires proving current Hydra doctrine, interfaces, staged assumptions, or carrier choices wrong, do that explicitly.

## Core task

Take the narrowed hidden-world lane from pass one and produce the strongest concrete design answer you can for actually improving Hydra strength, not just describing the target object.

This means you must output all of the following:

1. **Best Full Design**
   - the strongest full hidden-world stack you endorse, even if it exceeds current Hydra support
   - the exact teacher/search object, student object, runtime representation, and why this stack wins

2. **Shortest Honest Tranche**
   - the smallest buildable slice that preserves the semantics of the best full design instead of degenerating into a weaker fake approximation
   - if current Hydra support is too weak for the true winner, say so and define the smallest required contract migration

3. **Algorithm Family Decision Table**
   - compare the strongest surviving families and rank them
   - examples: CT-SMC world posterior, Rao-Blackwellized tile-world particle filtering, projected ensemble tracking, Sinkhorn-projected amortized posterior, latent roll-forward world model, oracle-guided blueprint + selective belief search, etc.
   - for each family, include:
     - exact object produced
     - whether it is student-facing, teacher-only, search-only, diagnostics-only, or forbidden
     - why it wins or loses on actual strength
     - what dependency or weakness kills it if it loses

4. **Teacher Hierarchy**
   - define the teacher stack clearly:
     - search-grade teacher
     - amortized student teacher
     - diagnostics-only objects
     - privileged/oracle objects
     - forbidden objects
   - make explicit what information each object is allowed to use and why

5. **Training Recipe**
   - the full recommended training loop for the winning design
   - include:
     - target generation path
     - losses/objectives
     - curriculum or tranche order
     - how oracle teachers or privileged information are used and then removed or distilled
     - how calibration and trust signals are trained or repaired
     - how this should interact with reward-side variance reduction rather than competing with it

6. **Evaluation Gates**
   - define the measurable conditions for promotion
   - include:
     - belief quality metrics
     - calibration metrics
     - action-quality / decision-delta metrics
     - arena / Elo-like or match metrics if appropriate
     - what counts as “real strength gain” instead of internal metric overfitting

7. **Kill Criteria**
   - define exactly what evidence would falsify the winning lane
   - do not protect your own recommendation with vague escape hatches
   - if the winning stack fails, specify the strongest fallback, not just “more research needed”

8. **Where Hydra Is Wrong**
   - explicitly list which current Hydra assumptions, staged decisions, interfaces, carriers, or doctrine claims should be overturned if they block the stronger design
   - this section is mandatory
   - do not hide behind “not currently supported” if the current support is the thing holding Hydra back

9. **What Stays Open**
   - list only the genuinely unresolved parts that remain after your best design decision
   - do not dump a lazy wish list here

## Hard rules

- Do **not** rerun the diagnosis or existence-check loop from pass one.
- Do **not** spend most of the answer on “what Hydra currently has.” Use current Hydra state only as build-cost evidence, not as the outer bound of the answer.
- Do **not** constrain yourself to “only what Hydra already supports.” The entire point of this pass is to find where Hydra must be changed to become stronger.
- Do **not** protect the current carrier, current head layout, current staging order, or current doctrine if they are inferior.
- Do **not** output a survey. Output a ranked verdict.
- Do **not** say several options are roughly tied unless you can prove they are tied under the same evidence and give a smallest decisive experiment matrix.
- Keep student-facing deployable objects public-information legal unless you explicitly label them as teacher-only or diagnostics-only.
- You may use privileged or oracle information for teachers, diagnostics, calibration, or bridge training, but you must say exactly how it is fenced.
- For every major recommendation, include:
  - why it improves Hydra strength,
  - how it would be validated,
  - what would falsify it.
- If a design depends on a stronger contract migration, state the migration explicitly instead of pretending the current contract is sufficient.
- If a recommendation is genie-only and not buildable right now, label it as genie-only. Then still provide the strongest honest tranche below it.

## Output shape requirements

Your answer must be organized under these exact top-level headings:

1. `Executive Verdict`
2. `Best Full Design`
3. `Shortest Honest Tranche`
4. `Algorithm Family Decision Table`
5. `Teacher Hierarchy`
6. `Training Recipe`
7. `Evaluation Gates`
8. `Kill Criteria`
9. `Where Hydra Is Wrong`
10. `What Stays Open`
11. `Minimal Experiment Matrix`

## Quality bar

Your answer fails if it does any of the following:

- spends more time diagnosing the lane than designing it
- refuses to overturn Hydra assumptions even when the stronger design requires it
- gives no primary winning design
- gives no explicit fallback
- gives no measurable promotion gates
- gives no kill criteria
- hides behind “not currently supported by Hydra” as a reason to avoid recommending the strongest design
- treats buildability as more important than eventual strength without stating that tradeoff openly

## Tone and detail

- Be decisive.
- Use formulas when needed.
- Use code-like detail when needed.
- Use tables when they actually help compare algorithm families or teacher objects.
- Make the validation path concrete enough that another strong agent could implement or falsify your design later.

## Final reminder

Pass one already asked what the hidden-world object should be.

Pass two is different.

Pass two asks:

**What exact design stack should Hydra build if the goal is actual strength gain, and where must Hydra be willing to be wrong to get there?**
