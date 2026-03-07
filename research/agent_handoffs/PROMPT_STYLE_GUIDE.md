# Hydra prompt style guide for long-horizon GPT-5.4 Pro sessions

This guide captures the prompt-writing patterns that currently work best for Hydra's external deep-work sessions.

It combines:
- OpenAI's GPT-5.4 prompt guidance
- what has worked in Hydra's long-horizon research and coding prompts
- what failed in earlier prompt iterations and should be avoided

Use this guide when writing or revising prompts for:
- external long-think research agents
- breakthrough / invention prompts
- repo-aware implementation prompts
- follow-up prompts that must converge to code-ready output

This is not a generic prompt-engineering guide. It is a Hydra-specific style guide for prompts that need strong architectural reading, disciplined retrieval, and coding-oriented synthesis.

---

## 1. Core design goals

Hydra prompts should push the model toward these behaviors:

1. Read the project holistically before narrowing.
2. Search broadly before converging.
3. Reject weak ideas instead of preserving them out of politeness.
4. Ground claims in retrieved evidence and real repo surfaces.
5. Return output that is exhaustive, high-signal, and ready for action.
6. End in a form that a coding agent can directly use.

In practice, the best Hydra prompts do **not** try to control every thought. They control:
- what context must be read first
- what counts as done
- what evidence is allowed
- what output shape is required

---

## 2. What the OpenAI guide reinforces

The official GPT-5.4 prompt guidance strongly matches what worked in Hydra.

Highest-value official themes:

- explicit output contracts improve reliability
- detailed, information-dense output works better than vague brevity when the task is technical
- tool persistence matters when correctness depends on retrieval
- dependency checks reduce skipped prerequisite reading
- long-horizon tasks need explicit completeness rules
- research prompts work better with staged retrieval and synthesis
- grounding and citation rules reduce unsupported claims
- verification loops improve final quality on complex work

Official source:
- https://developers.openai.com/api/docs/guides/prompt-guidance/
- use this together with the GPT-5.4-specific local notes in `/home/nikketryhard/dev/hydra/prompting.md`

Related Hydra-local source:
- `/home/nikketryhard/dev/hydra/prompting.md`

---

## 3. The Hydra prompt voice

Hydra prompts work best when they sound like this:

- direct
- technical
- disciplined
- high-standard
- anti-handwave
- anti-fluff
- willing to reject weak ideas

### Verbosity default for technical Hydra prompts

For deep technical prompts, Hydra should now default to more detail, not less.

That means prompts should usually ask for:

- multi-paragraph explanations instead of label-only fragments
- explicit mechanism walkthroughs instead of compressed summaries
- code snippets, pseudocode, API sketches, and data-structure examples
- equations when the recommendation depends on scoring, optimization, uncertainty, or probability
- full references and source links for every serious claim
- implementation caveats, failure modes, thresholds, assumptions, and tradeoffs

Do not accidentally compress a technically rich answer into a thin executive summary.

If the task is research-heavy, math-heavy, or architecture-heavy, the prompt should bias toward:

- verbose reasoning
- explicit derivations where useful
- worked examples
- benchmark detail
- enough depth that a coding agent can implement without reconstructing missing logic

Good phrasing:

- "Be as detailed and explicit as necessary; do not optimize for brevity."
- "Return a full technical treatment, not a compressed memo."
- "Include equations, derivations, and worked examples where they materially improve correctness."
- "Include code snippets or pseudocode for every major mechanism."
- "Cite all sources and attach references to concrete claims."
- "Prefer full technical exposition over compressed summary."

Bad phrasing:

- "Keep the answer compact" when the task is deep or technical
- "Be concise" with no guardrails on omitted mechanism
- "Summarize briefly" when we actually need a prototype-ready answer
- "High level only" when exact insertion points, formulas, or code shape matter

Good Hydra-style phrases:

- "Do not give another broad survey."
- "Read the core docs holistically first."
- "Do not stop at the first plausible answer."
- "Reject weak mappings explicitly."
- "Do not preserve a proposal just because it is interesting."
- "Ground every serious proposal in exact files/functions/structs likely to change."
- "Success means a coding agent could start from your answer with minimal guesswork."

Bad tone patterns:

- hypey futurism
- generic "brainstorm 10 ideas"
- overly conversational filler
- soft indecisive ranking with no final recommendation
- vague "best practices" with no repo grounding

---

## 4. Required structural blocks

For long-horizon Hydra prompts, these blocks are the default backbone.

### Required in most prompts

```xml
<output_contract>
<verbosity_controls>
<research_mode>
<tool_persistence_rules>
<calculation_validation_rules>
<dependency_checks>
<completeness_contract>
<citation_rules>
<grounding_rules>
<verification_loop>
```

### Add when the failure mode is common

```xml
<empty_result_recovery>
<dig_deeper_nudge>
```

### What verbosity controls should do now

For Hydra's serious technical prompts, `<verbosity_controls>` should usually prevent vagueness, not enforce shortness.

Good defaults:

```xml
<verbosity_controls>
- Be as detailed and explicit as necessary; do not optimize for brevity.
- Prefer full technical exposition over compressed summary.
- Do not omit equations, derivations, thresholds, assumptions, edge cases, or implementation caveats when they matter.
- Use multi-paragraph explanations when a short paragraph would hide important logic.
</verbosity_controls>
```

### Block intent by prompt type

- **follow-up engineering prompt**
  - stronger dependency checks
  - stronger code-grounding requirements
  - explicit file/function insertion-point requirements

- **fresh-context research prompt**
  - stronger holistic-ingestion rules
  - stronger narrowing workflow
  - stronger candidate-pruning and kill-criteria requirements

- **invention prompt**
  - stronger anti-memo pressure
  - stronger prototype path / benchmark / kill criteria requirements

---

## 5. The most important Hydra lesson: holistic ingestion first

This has been one of the biggest real-world wins.

Older agents often behaved like sysadmins reading logs:
- grep a keyword
- inspect 40 lines around it
- jump to a conclusion

That behavior is terrible for Hydra architecture work because the meaning of the design docs is spread across:
- doctrine sections
- caveats
- sequencing notes
- active-vs-reserve distinctions
- interactions between runtime/search/training systems

So for Hydra prompts, whole-document reading is mandatory for core docs.

### Use this rule explicitly

```xml
<holistic_ingestion_rules>
- Read the core docs as whole documents before narrowing.
- Do not start with keyword search on the core docs.
- Do not rely on fragmented line-window retrieval for architecture understanding.
- After holistic reading, you may use targeted search for exact details.
</holistic_ingestion_rules>
```

### Best reading order pattern

For most serious Hydra prompts:

1. `research/design/HYDRA_RECONCILIATION.md`
2. `research/design/HYDRA_FINAL.md`
3. `docs/GAME_ENGINE.md`
4. `research/design/OPPONENT_MODELING.md` when relevant
5. prior answer anchors (`ANSWER_1-1.md`, `ANSWER_2-1.md`, `ANSWER_3-1.md`)
6. exact code-grounding files
7. outside papers / GitHub examples

That order matters because it prevents the model from inventing against stale or partial project context.

---

## 6. Browse-first is now better than zip-first for core docs

Hydra originally leaned heavily on zip attachments.

That still helps in some environments, but for very long docs the better default is now:

- raw GitHub Markdown links for the core architectural docs
- browse/fetch tool for holistic reading
- targeted narrowing only after full-document ingestion

Why:
- some agents fragment zip/file reading into tiny chunks
- raw GitHub markdown often works better with browse tools for whole-doc reading
- the model wastes less effort navigating package structure

### Current recommended source-access pattern

For browse-capable models:

- primary source material = raw GitHub links
- code grounding = raw GitHub source links
- outside evidence = official papers / docs / GitHub examples retrieved during the session

Do not write prompts that say:
- "Use raw links only if the attachment fails"

For Hydra's current long-horizon prompt style, raw-link browsing is often the better first-class path.

---

## 7. Research workflow that works

The best Hydra prompts use a staged workflow.

### Recommended pattern

```xml
<research_mode>
- Work in 3 passes:
  1. Ingest: read the core Hydra docs holistically first.
  2. Retrieve: identify the unresolved questions, then search broadly and follow 1-2 strong second-order leads.
  3. Synthesize: reject weak branches and converge to a small number of serious recommendations.
- Stop only when more searching is unlikely to materially change the conclusion.
</research_mode>
```

### Practical interpretation

Pass 1: understand Hydra as it actually exists now  
Pass 2: search outside fields and repo surfaces  
Pass 3: produce an answer that can survive contact with coding reality

Do not let the model jump straight from one interesting paper to a final answer.

---

## 8. Tool behavior rules that help a lot

These rules matter a lot in long sessions.

### Good defaults

```xml
<tool_persistence_rules>
- Use tools whenever they materially improve correctness, completeness, or grounding.
- Do not stop early when another tool call is likely to materially improve correctness or completeness.
- Keep calling tools until the task is complete and verification passes.
- If a tool returns empty or partial results, retry with a different strategy.
- When the answer depends on non-trivial arithmetic, statistics, simulation, or parameter tradeoffs, use Python from bash to calculate or sanity-check the result instead of relying on mental math.
- Prefer short Python scripts in bash for complex calculations, quick numerical experiments, equation checks, threshold sweeps, and counterexample hunting.
</tool_persistence_rules>
```

```xml
<calculation_validation_rules>
- If the answer depends on a complex calculation, use Python in bash to compute or sanity-check it.
- Use short Python scripts for parameter sweeps, expected-value checks, uncertainty calculations, simulation sanity checks, or numerical comparisons.
- Report the relevant computed result in the final answer when it materially supports the recommendation.
- Do not fake arithmetic that could have been verified with a quick script.
</calculation_validation_rules>
```

```xml
<dependency_checks>
- Before taking an action, check whether prerequisite discovery, lookup, or memory retrieval steps are required.
- Do not skip prerequisite steps just because the intended final action seems obvious.
- If the task depends on the output of a prior step, resolve that dependency first.
</dependency_checks>
```

### Hydra-specific addition

For architecture-heavy prompts, add this idea explicitly:

- broad reading before narrow search
- synthesis pause after retrieval
- then final recommendation

The model should not retrieve forever. It should retrieve hard, then converge.

---

## 9. What counts as a good final answer

Hydra prompt outputs should usually end in one of these forms:

### A. Engineering brief

- what Hydra already has
- what the new idea adds
- exact mechanism
- exact file/function insertion points
- needed labels/signals/logs
- pseudocode / API sketch
- code snippets where the mechanism would otherwise stay vague
- equations or derivations where the recommendation depends on math
- worked examples or sample calculations where useful
- full references and exact citations
- benchmark and kill criteria

### B. Strategic cut

- what to keep
- what to demote
- what to archive
- what to try first
- why weaker paths lose

### C. Invention/prototype recommendation

- 1-3 serious candidates max
- mechanism, not slogan
- why it fits Hydra specifically
- why it clears the separator bar rather than being merely incremental
- closest known baseline and why it does not reduce to it
- dependency closure table for required signals, labels, hooks, teacher outputs, and runtime state
- cheapest prototype path
- minimum falsifiable prototype
- what would falsify it quickly

For breakthrough-oriented prompts, also require:
- a short `Hydra posture reconstruction` before ideas
- the strongest simpler mainline alternative for each candidate
- explicit rejected directions, not just surviving finalists
- an allowed `0 surviving candidates` outcome if nothing really survives

If the output reads like a literature survey, the prompt underperformed.

---

## 10. Always force pruning

Hydra prompts get much stronger when they force the model to prune.

Good pruning language:

- "Return only 1-3 serious candidates."
- "Reject weak mappings explicitly."
- "Do not preserve a proposal just because it is interesting."
- "End with the single best candidate to try first."
- "Give the single best cheap benchmark to run first."
- "If no candidate survives, return 0 surviving candidates and explain why."

Without pruning pressure, long-horizon prompts tend to degrade into:
- sprawling rankings
- pseudo-comprehensive but weak idea lists
- no decisive next step

---

## 11. Always force grounding

Hydra-specific grounding is non-negotiable.

Good grounding rules:

```xml
<citation_rules>
- Cite only sources actually retrieved in the current workflow.
- Never fabricate references.
- Attach citations to the exact claims they support.
- Include full reference detail and direct source links when possible.
</citation_rules>
```

```xml
<grounding_rules>
- Base repo claims only on the repo docs/code or retrieved raw links.
- If a statement is an inference rather than directly supported, label it as an inference.
- If sources conflict, state the conflict explicitly.
</grounding_rules>
```

Hydra prompts should also usually require:
- exact file names
- exact functions or structs if known
- explicit note when a needed signal or label does not yet exist
- explicit marking of any unevidenced repo surface as `inference` or `[blocked]`
- explicit equations or formulas when claims depend on quantitative reasoning
- explicit code snippets, pseudocode, or API sketches for all major mechanisms
- enough source detail that a later agent can retrace the evidence without guessing
- explicit dependency closure when implementation depends on new labels, hooks, trajectories, or teacher outputs

For breakthrough/invention prompts, good extra grounding pressure is:

```xml
<posture_reconstruction_rules>
- Before proposing ideas, include a short "Hydra posture reconstruction" section with 5-10 bullets.
- Distinguish current mainline doctrine, reserve-shelf ideas, partially closed loops, and non-goals/deprioritized paths.
- Do not propose breakthrough candidates until that posture reconstruction is complete.
</posture_reconstruction_rules>
```

```xml
<novelty_honesty_rules>
- For every surviving candidate, include a "closest known baseline" subsection.
- State the nearest known method or family, the exact overlap, and the irreducible difference.
- If the method reduces to a known technique under realistic Hydra constraints, downgrade or reject it.
- Label each candidate as:
  - `A`: genuinely new mechanism
  - `B`: known mechanism with a Hydra-specific adaptation that plausibly changes capability
  - `C`: renamed or lightly modified known trick
- Reject all `C` candidates.
</novelty_honesty_rules>
```

```xml
<minimum_falsification_rules>
- For every surviving candidate, define the minimum falsifiable prototype that tests the claimed mechanism in isolation.
- If the core claim cannot be tested without a large coupled rollout or major stack build-out, reject the idea as too diffuse.
- The first benchmark should distinguish the idea from stronger tuning, more search, more data, or easier teacher signals.
</minimum_falsification_rules>
```

```xml
<abstention_rules>
- If evidence is missing, conflicting, or too weak, output `insufficient evidence` instead of filling gaps by inference.
- Unsupported claims must not appear in the final recommendation unless clearly labeled as a hypothesis plus falsification path.
- If no candidate survives the novelty, grounding, and prototypeability filters, return `0 surviving candidates` and explain why.
</abstention_rules>
```

```xml
<claim_verification_rules>
- Before finalizing, list the highest-leverage factual or technical claims in the draft.
- Generate verification questions for those claims and answer them independently using retrieved evidence.
- Revise the answer using those verification results instead of relying on first-pass confidence.
</claim_verification_rules>
```

```xml
<evidence_bucket_rules>
- Separate important statements into `supported`, `inference`, and `hypothesis`.
- Only `supported` and clearly labeled `inference` may influence the final recommendation.
- `hypothesis` items must come with a falsification path and may not be presented as established facts.
</evidence_bucket_rules>
```

```xml
<feasibility_scorecard_rules>
- For each surviving candidate, score `novelty`, `feasibility in Hydra`, `evidence strength`, and `cheap-testability` on a fixed scale.
- If novelty exceeds feasibility by a large margin and no decisive cheap test exists, reject or downgrade the idea.
</feasibility_scorecard_rules>
```

---

## 12. Always force completion and verification

This is another area where the official GPT-5.4 guidance and Hydra experience line up hard.

Use:

```xml
<completeness_contract>
- Treat the task as incomplete until all requested deliverables are covered or explicitly marked [blocked].
- Mark underspecified items [blocked] rather than pretending they are ready.
</completeness_contract>
```

```xml
<verification_loop>
- Before finalizing:
  - check correctness
  - check grounding
  - check format
  - check whether the recommendation is actionable
</verification_loop>
```

Hydra-specific upgrade:

- add a final check that the model actually read the core docs holistically before narrowing in
- add a final check that each surviving candidate beats the strongest simpler mainline alternative on more than vibes
- add a final check that each surviving candidate has a minimum falsifiable prototype
- add a final check that no candidate survived only because the prompt implicitly demanded at least one answer
- add a final check that the strongest claims were independently verified rather than merely repeated confidently

That one catches a huge amount of fake understanding.

---

## 13. Recommended prompt skeleton

This is the default Hydra skeleton for long-horizon research/coding prompts.

```text
# Title

Short task framing.

Primary source material lives in the raw GitHub links below.

## Critical directive — how to read the core Hydra docs
<holistic_ingestion_rules>
...
</holistic_ingestion_rules>

## Reading order
1. core doctrine docs
2. prior answer anchors
3. code-grounding files
4. outside retrieval

## Raw GitHub links
- ...

Task body

<output_contract>
...
</output_contract>

<verbosity_controls>
...
</verbosity_controls>

<calculation_validation_rules>
...
</calculation_validation_rules>

<research_mode>
...
</research_mode>

<tool_persistence_rules>
...
</tool_persistence_rules>

<dependency_checks>
...
</dependency_checks>

<posture_reconstruction_rules>
...
</posture_reconstruction_rules>

<completeness_contract>
...
</completeness_contract>

<citation_rules>
...
</citation_rules>

<grounding_rules>
...
</grounding_rules>

<abstention_rules>
...
</abstention_rules>

<claim_verification_rules>
...
</claim_verification_rules>

<evidence_bucket_rules>
...
</evidence_bucket_rules>

<novelty_honesty_rules>
...
</novelty_honesty_rules>

<minimum_falsification_rules>
...
</minimum_falsification_rules>

<feasibility_scorecard_rules>
...
</feasibility_scorecard_rules>

<verification_loop>
...
</verification_loop>

<dig_deeper_nudge>
...
</dig_deeper_nudge>
```

---

## 14. Anti-patterns to avoid

Do not write Hydra prompts that do these things:

1. **zip-first by default** when browse/raw reading is available and the docs are long
2. **search-first reading** of core architecture docs
3. **broad survey output** with no forced convergence
4. **no kill criteria** for new ideas
5. **no file-level grounding** for engineering recommendations
6. **no explicit completion rules**
7. **no explicit evidence boundary**
8. **too many candidates** kept alive at once
9. **generic "be creative" phrasing** without prototype pressure
10. **architectural recommendations before doctrine ingestion**
11. **over-compressing technical answers** until code, formulas, assumptions, or derivations disappear
12. **hand-wavy arithmetic** that should have been checked with Python
13. **breakthrough framing with no separator bar**
14. **novelty claims with no closest-baseline comparison**
15. **implementation claims with no dependency-closure check**
16. **prototype claims with no minimum falsifiable core**
17. **forcing at least one idea to survive even when none clear the bar**
18. **confident recommendation with no abstention path when evidence is weak**
19. **important claims left unverified because the prompt only asks for a vague final self-check**
20. **speculation blended into grounded claims with no evidence buckets**

---

## 15. Current Hydra-specific best practices

Based on the prompt iterations that worked best so far:

1. Use browse/raw links for core docs.
2. Force whole-doc reading before narrowing.
3. For serious technical tasks, prefer detailed, high-density output over compressed summaries.
4. Demand exact insertion points and benchmark plans.
5. Force candidate pruning.
6. Force kill criteria.
7. Treat stale doctrine as a real hazard and re-anchor on reconciliation first.
8. For follow-up prompts, say "do not give another broad survey."
9. For invention prompts, say "do not settle for a shallow invention memo."
10. For implementation prompts, say "success means a coding agent could begin prototyping from your answer with minimal guesswork."
11. Explicitly request code snippets, equations, worked examples, and exhaustive references when technical depth matters.
12. Explicitly allow Python in bash for complex calculations and ask the agent to report computed results when relevant.
13. For breakthrough prompts, require a separator-bar test: why this changes ceiling rather than merely improving slope, stability, or convenience.
14. Require a closest-known-baseline section to kill fake novelty.
15. Require a dependency-closure table for signals, labels, hooks, trajectories, teacher outputs, and runtime state.
16. Require a minimum falsifiable prototype, not just a broad prototype path.
17. Permit `0 surviving candidates` when novelty, grounding, or prototypeability filters kill everything.
18. Require a short list of rejected directions so the model shows selection pressure instead of polished survey drift.
19. Add an abstention rule so weak evidence yields `insufficient evidence`, not polished bluffing.
20. For hard prompts, require a claim-level verification pass, not just a generic final self-check.
21. Use evidence buckets (`supported`, `inference`, `hypothesis`) when the answer mixes repo facts, external evidence, and synthesis.
22. Add a novelty-vs-feasibility scorecard when the task risks rewarding coolness over prototype realism.

---

## 16. Where to use this guide

Use this file when editing:
- `PROMPT_5_FOLLOWUP_COMPUTE_ROUTER_AND_ROBUSTNESS.md`
- `PROMPT_6_FOLLOWUP_DEBC_AR.md`
- `PROMPT_FRESH_CONTEXT_CROSS_FIELD_BREAKTHROUGH.md`
- `PROMPT_FRESH_CONTEXT_INVENT_NEW_TECHNIQUES.md`
- `prompt-5.md`
- `prompt-6.md`
- future files in `research/agent_handoffs/prompts/`

This guide is especially meant for:
- browse-first raw-link prompts
- follow-up prompts that must converge to code-ready detail
- fresh-context prompts that must reconstruct Hydra quickly without drifting into shallow survey mode

If a future prompt deviates from this guide, it should be because the task shape truly changed, not because the writer forgot the lessons.
