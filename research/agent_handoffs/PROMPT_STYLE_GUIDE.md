WARNING! You do not need to write the artifacts on your own since normally prompts are around 8000 lines and that is too much. Use the prompting tool at ### 12.1 Tool location of this document to learn how to prompt with the tool.

# Hydra Prompt Style Guide — Artifact-First Batch Prompting Doctrine

This file is the doctrine for writing Hydra’s new batch prompts.

It is not a giant prompt dossier.
It is not where we dump 1000+ lines of code and paper artifacts.
Those belong inside the actual prompts sent to agents.

This guide should stay concise enough that an LLM or human can read it quickly and know exactly what to do.

The whole point is simple:

- prompts should be short at the top
- prompts should be artifact-heavy in the body
- artifacts should be treated as evidence, not truth
- the answer should be a blueprint, not a memo

---

## 1. Core doctrine

### 1.1 The answer must be a blueprint

Do not ask for:

- a memo
- a broad survey
- a high-level recommendation note
- a polished strategy essay

Ask for:

- an implementation-ready blueprint
- a validation-ready blueprint
- or a risk-audit blueprint

The answer should help the next engineer or reviewer act, not just admire the prose.

### 1.2 The prompt should carry the starting evidence

Do not make the agent do all the first-mile rediscovery from nothing.

The prompt should already contain:

- low-level code excerpts
- doc excerpts
- tests
- structs
- formulas
- thresholds
- comments
- examples

The point is not to pre-solve the task.
The point is to give the agent a dense starting packet it can critique, validate, and build from.

### 1.3 The artifacts are not truth

The prompt must explicitly say the artifacts are only what the current codebase/docs appear to say.
They may be:

- stale
- partial
- inconsistent
- semantically wrong
- misleading by omission

The agent must treat them as evidence to inspect and critique, not truth to inherit.

### 1.4 Narrowness is good by default

Most prompts should be narrow.

That is good for:

- Hand-EV repair
- target provenance decisions
- tiny ponder scorers
- rollout disable policy
- conservative ExIt and delta-q validation

Do not loosen a narrow implementation prompt into a broad invention prompt by accident.

### 1.5 The model must not stop early

The prompt should explicitly force:

- discovery
- thinking
- testing
- validation
- repeated looping until saturation or blockage

We use aggressive looping language because the old failure mode was premature finish with weak evidence.

### 1.6 No dump in any logic

The model must not hide important reasoning in black-box conclusions or polished assertions that cannot be reconstructed from the evidence packet or the explicit blueprint it writes.

Every important mechanism, threshold, recommendation, or architecture move should be either:

- directly inferable from the supplied artifacts
- supported by cited external evidence
- or made explicit inside the blueprint with enough visible derivation that we can validate and reproduce it ourselves

If the logic cannot be retraced, it should not be presented as settled.

---

## 2. Default top-of-prompt shell

Use this shell for most serious Hydra prompts.

```xml
<role>
Produce an implementation-ready blueprint.
Do not give a memo.
Your answer itself must be the blueprint.
</role>

<direction>
Work toward the strongest exact blueprint for [TASK].

We want a detailed answer that makes clear:
- [decision point 1]
- [decision point 2]
- [decision point 3]
- [what must stay narrow / deferred / rejected]
- [how to implement or validate the surviving path with minimal guesswork]

Use the artifacts below to derive your conclusions.
</direction>

<style>
- no high-level survey
- no vague answer
- include reasoning
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail that we can validate it ourselves (pdfs, sources, links, similar projects)
- distinguish direct artifact support from your own inference
- use search/browse to find the original paper, then inspect the full PDF with skill; use abstracts or summaries only for discovery, not as the final evidence base
- use the bash tool to run Python for calculations, math checks, and validation when rigor matters
- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked, and do not stop before at least 20+ such loops (as much loops as possible tho)
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.
</artifact_note>

<artifacts>
...
</artifacts>
```

The top shell should stay lean.
The bulk should be the artifacts.

---

## 3. What changed from the old doctrine

Old style:

- read-order heavy
- raw-link heavy
- meta-heavy
- lots of output-contract machinery
- easy to turn into a polished audit memo

New style:

- short role / direction / style shell
- explicit artifact skepticism
- artifact-heavy body
- blueprint answer
- PDF-first paper handling
- more visible reasoning, less inherited framing

---

## 4. What prompts must force

### 4.1 Blueprint over memo

The answer should feel buildable or directly auditable.

### 4.2 Anti-vagueness

Punish these failure modes:

- one-paragraph summaries
- broad “best practice” talk
- implementation claims with no mechanism
- benchmark claims with no metrics
- reasoning with no formulas where formulas matter

### 4.3 Evidence buckets

The answer should separate:

- direct artifact support
- external source support
- inference
- proposal
- blocked / missing surface

When confidence is high, the answer must also show why that confidence level is warranted.
Do not let a confident tone do the work of evidence.

For every major recommendation, architecture move, threshold, or implementation claim, the answer should make visible:

- what directly supports it
- what is inferred rather than directly supported
- why the inference is still justified
- how a reviewer could validate or falsify it later

If those pieces are missing, the claim should be downgraded to a weaker confidence label.

### 4.4 External paper discipline

When papers matter:

1. find the original paper
2. inspect the full PDF
3. use abstracts only for discovery
4. say when support is direct vs analogy

### 4.5 Math and calculation rigor

If the task touches:

- parameter count
- compute budget
- drift thresholds
- expected value
- throughput
- weighting
- confidence intervals
- probabilities

then the prompt should explicitly allow Python in bash for validation.

### 4.6 Saturation before finish

The prompt should make it hard for the model to stop after the first plausible answer.

### 4.7 Confidence requires justification and validation

Prompts should force the model to justify confidence, not just conclusions.

If the answer claims that something works, survives, is the best plan, or is implementation-ready, it should also explain:

1. why that claim is warranted by the evidence bucket it belongs to
2. how that claim could be validated later by a reviewer or implementer
3. what evidence would have to appear for the claim to be downgraded or killed

This matters most for architecture choices, file-level plans, thresholds, and “smallest decisive unblocker” claims.

The model should actively inspect its own draft and ask:

- can I point to enough evidence for this confidence level?
- is this directly supported, or am I silently upgrading a proposal into a settled plan?
- could another reviewer reproduce or falsify this conclusion from the visible reasoning?

If the answer cannot pass that self-check, it should relabel the claim as inference, proposal, or blocked.

---

## 5. What prompts must not force

Do not over-constrain output structure so hard that the model becomes a bureaucrat.

The guide should stay concise, but actual prompts should still constrain output style enough to keep the model focused on a few concrete topics. The goal is not zero structure. The goal is enough structure to keep the answer narrow and useful without turning it into a rigid report generator.

Avoid prompts that require:

- twenty named sections in exact order
- endless claim ledgers for narrow tasks
- giant “prove you followed the prompt” output rituals
- template prison instead of actual reasoning

The right balance is:

- strong role
- clear direction
- anti-vagueness rules
- artifact skepticism
- artifact-heavy body
- enough output-style constraint to keep the answer focused on the actual lane

---

## 6. How much explanation should sit above the artifacts

Very little.

Good:

- role
- direction
- style
- artifact note

Sometimes:

- one small scope note
- one small novelty rule

Bad:

- giant system-context sermons
- “important truths” sections that pre-solve the task
- narrative walkthroughs of what the artifacts already say

The prompt should not explain the evidence too much before the model has looked at it.

---

## 7. Novelty rules

The new prompt style is mostly implementation-first and anti-drift.
That means most prompts should not encourage broad novelty.

### Good places for novelty clauses

- long-run risk audits
- architecture reserve-shelf prompts
- breakthrough prompts
- prompts explicitly asking whether a stronger adjacent-field formulation exists

### Bad places for novelty clauses

- Hand-EV repair prompt
- narrow provenance decision prompt
- tiny ponder scorer prompt
- rollout disable-policy prompt

### Default novelty clause

Use this when you want bounded adjacent-field exploration:

```xml
- after grounding in the artifacts, actively search adjacent fields for stronger alternative formulations of the same problem; keep them only if they survive validation against the artifacts
```

### Strong novelty add-on

Use this only when you explicitly want broader cross-field synthesis:

```xml
- after grounding in the artifacts, explore many adjacent fields for competing formulations of the same problem, keep searching for interesting fragments worth fusing together, and continue the explore -> think hard -> validate loop until the strongest fused formulation either survives or is killed by the artifact constraints
```

Do not add novelty clauses everywhere.

---

## 8. The right stance toward references

We do not want:

- pretty bibliography dressing
- one-line paper name drops
- abstract-only support
- “paper X vibes like this idea” with no mechanism

We do want:

- actual formulas
- actual method-defining wording
- actual scope limits
- actual caveats
- actual places where the paper’s theory breaks or narrows

Good prompt references should include enough context to understand why the source matters.

---

## 9. Reusable wording patterns

### Role block

```xml
<role>
Produce an implementation-ready blueprint.
Do not give a memo.
Your answer itself must be the blueprint.
</role>
```

### Artifact skepticism block

```xml
<artifact_note>
The artifacts below reflect what the current codebase/docs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.
</artifact_note>
```

### Paper handling line

```xml
- use search/browse to find the original paper, then inspect the full PDF with skill; use abstracts or summaries only for discovery, not as the final evidence base
```

### Math rigor line

```xml
- use the bash tool to run Python for calculations, math checks, and validation when rigor matters
```

### Anti-premature-stop line

```xml
- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked, and do not stop before at least 20+ such loops (as much loops as possible tho)
```

### Anti-vagueness core

```xml
- no high-level survey
- no vague answer
- include reasoning
- when you sound confident, show the justification for that confidence level
- for every important claim, make the validation path visible enough that a reviewer can test it later
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail that we can validate it ourselves (pdfs, sources, links, similar projects)
- distinguish direct artifact support from your own inference
- do not dump logic; every important mechanism, threshold, or recommendation should be inferable from evidence or made explicit in the blueprint so it can be validated and reproduced
```

### Confidence-and-validation line

```xml
- if you claim a path works, survives, or is implementation-ready, show why that confidence is justified and how the claim can be validated or falsified later
- inspect your own draft before finishing: if a confident claim is not objectively justified by visible evidence, downgrade it to inference, proposal, or blocked
```

---

## 10. Failure modes to ban explicitly

### Failure mode 1 — Memo mode

Symptoms:

- polished prose
- broad conclusions
- weak mechanism detail
- feels smart but cannot be built from

Ban with:

- “do not give a memo”
- “your answer itself must be the blueprint”

### Failure mode 2 — Truth inheritance

Symptoms:

- model repeats stale docs confidently
- model treats artifact dump as ground truth
- model fails to critique semantics

Ban with:

- artifact note
- direct-support vs inference distinction

### Failure mode 3 — Abstract citation theater

Symptoms:

- references look impressive
- support is shallow
- full paper never checked

Ban with:

- PDF-first rule
- direct vs analogy support requirement

### Failure mode 4 — Parameterized overconfidence

Symptoms:

- exact thresholds appear with no calibration
- exact constants sound authoritative without evidence

Ban with:

- Python-in-bash validation rule
- explicit support vs proposal distinction

### Failure mode 5 — Premature convergence

Symptoms:

- first plausible idea wins
- no second-order search
- no falsification

Ban with:

- long loop rule
- saturation / blockage rule

### Failure mode 6 — Survey drift

Symptoms:

- answers list many possibilities
- no final blueprint
- no pruning

Ban with:

- blueprint framing
- narrow direction bullets

### Failure mode 7 — Typed-hole confusion

Symptoms:

- model sees a head/field and assumes it is live, trained, or credible

Ban with:

- force artifact critique
- make “typed surface vs active target vs semantically valid object” a central distinction

### Failure mode 8 — Logic dump / black-box rigor theater

Symptoms:

- major decisions appear with no reconstructable reasoning path
- the answer sounds rigorous but the rigor lives only in hidden reasoning
- thresholds or architecture moves are asserted without enough visible support

Ban with:

- explicit anti-dump-in-logic rule
- require that important logic be inferable from artifacts or explicit blueprint content
- require enough visible derivation that a reviewer can reproduce the conclusion

### Failure mode 9 — Confidence without proof of confidence level

Symptoms:

- answer sounds highly certain, but the confidence level is not justified
- proposal-level architecture is written like settled doctrine
- validation plan exists, but justification for why the plan itself is the right one is weak

Ban with:

- require justification for confidence, not just the claim
- require visible validation or falsification path for every major confident statement
- require the model to self-audit its own draft and downgrade unsupported certainty

---

## 11. Reference example prompts

Use these archive examples as the canonical naming and shape references for the new prompt style:

- `reference_prompt_example_001_narrow_focused.md`
- `reference_prompt_example_002_broad_novel_fuse_loop.md`
- `reference_prompt_example_003_balanced_narrow_not_overconstrained.md`

These are examples of the current artifact-first doctrine. New example prompts added to the archive should follow this naming family.

Use them like this:

- narrow implementation / validation prompt -> `reference_prompt_example_001_narrow_focused.md`
- broad novelty / cross-field fusion prompt -> `reference_prompt_example_002_broad_novel_fuse_loop.md`
- narrow but not over-constrained prompt -> `reference_prompt_example_003_balanced_narrow_not_overconstrained.md`

---

## 12. Prompt generator tool

For repeated prompt authoring, use `scripts/generate_prompt.py` instead of hand-assembling every long prompt from scratch.

The tool is not a prompt framework.
It is a small JSON-driven utility for generating Hydra-style artifact-first prompts faster and more consistently.

Use it when:

- you want multiple prompt variants from one shared artifact packet
- you want reusable shell blocks like `role`, `direction`, `style`, and `artifact_note`
- you want line-ranged code/doc excerpts without manual copy-paste
- you want per-artifact labels and explanations so the artifact body has useful context
- you want to regenerate prompts quickly after changing the artifact set

Do not use it as an excuse to stop thinking about prompt quality.
The generator speeds up assembly.
It does not decide what artifacts belong in the prompt.

### 12.1 Tool location

- script: `scripts/generate_prompt.py`
- example config: `scripts/examples/prompt_config.example.json`
- tests: `scripts/tests/test_generate_prompt.py`

### 12.2 What the generator supports

The generator currently supports:

- multiple named variants in one config
- shared default shell sections and shared default artifacts
- per-variant shell overrides by tag
- per-variant artifact references
- per-variant one-off inline artifacts
- artifact labels
- artifact explanations
- source labels for line-number prefixes
- configurable fence language
- optional line numbering

Supported artifact types:

- `file_range` -> include a file excerpt with inclusive `start_line` / `end_line`
- `file_full` -> include a whole file
- `literal` -> include literal text from the config itself

This is enough for most Hydra prompt work.
Keep the tool simple unless real author pain proves otherwise.

### 12.3 Config shape

The config is JSON and follows a simple pattern:

- top-level `defaults` for shared shell sections and shared artifacts
- top-level `artifacts` registry for reusable artifact definitions
- `variants` for prompt-specific direction blocks and extra artifacts

The shell is built from tagged sections such as:

- `role`
- `direction`
- `style`
- `artifact_note`

Those tags are rendered as XML-style blocks in the output prompt.

Artifact entries can carry:

- `label`
- `explanation`
- `source_label`
- `fence_language`
- `show_line_numbers`

That means the generated artifact body can say what the artifact is, where it came from, and why it matters, instead of dumping bare snippets.

### 12.4 Minimal workflow

Typical workflow:

1. copy `scripts/examples/prompt_config.example.json`
2. define the shared shell sections you want in `defaults`
3. add reusable artifacts to the top-level `artifacts` array
4. add one or more `variants`
5. validate the config
6. generate one prompt or all variants
7. inspect the rendered prompt before sending it to an agent

Useful commands:

```bash
python3 scripts/generate_prompt.py --config scripts/examples/prompt_config.example.json --list-variants
python3 scripts/generate_prompt.py --config scripts/examples/prompt_config.example.json --validate-only
python3 scripts/generate_prompt.py --config scripts/examples/prompt_config.example.json --variant narrow-focused
python3 scripts/generate_prompt.py --config scripts/examples/prompt_config.example.json --all-variants --output-dir /tmp/hydra-generated-prompts
```

### 12.5 Authoring rules when using the generator

The same doctrine still applies:

- keep the shell lean
- let the body carry the evidence
- treat artifacts as evidence, not truth
- prefer dense excerpts over decorative links
- add explanation around artifacts when the context would otherwise be unclear
- do not bulk up prompts with filler just because the generator makes it easy

Bad generated prompt:

- huge because of `file_full` spam
- many artifacts with no explanation of why they matter
- variants that differ only cosmetically
- copied doctrine blocks but weak task-specific artifacts

Good generated prompt:

- short shell
- dense artifact packet
- enough explanation to orient the reader
- variant-specific direction only where it meaningfully changes the task

### 12.6 What the generator does not do

The generator does not:

- decide the right task framing for you
- guarantee the prompt is narrow enough
- guarantee the prompt is long enough for hard tasks
- guarantee the artifacts are semantically correct
- replace manual review of the final rendered prompt

Always inspect the generated output before using it.

---

## 13. Long-prompt rule

The guide does not need to be huge.
The prompts should be.

For serious external-agent work, default to prompts that are at least 2000 lines unless the task is genuinely too small to justify that much context.

Those 2000+ lines should come from a real context dump, including:

- real code
- real docs
- real tests
- real formulas
- real comments
- real thresholds
- references and quotes directly pulled from papers/PDFs
- the method-defining words and formulas we actually use
- enough surrounding context around every artifact block that the model can critique it instead of just pattern-matching on isolated lines

Do not use links as the main evidence body inside prompts. Use the words, formulas, snippets, and context itself.

Long prompts should be built like this:

- real code
- real docs
- real tests
- real formulas
- real paper/PDF snippets with context
- real constraints
- useful context before each artifact block

The artifact body should not be a pile of one-line texts with no context.
Each artifact block should have enough title/comment/context around it that the model understands what it is looking at and why it matters.

Do not bulk it up with:

- one-line slogans
- repeated meta-rules
- broad filler prose
- fake code

The guide should stay concise.
The prompts carry the bulk.

---

## 14. Final checklist for prompt writers

Before shipping a prompt, check:

- role is blueprint-first
- direction is narrow and concrete
- style includes anti-vagueness, PDF-first, Python-in-bash, and anti-premature-stop rules
- artifact note says artifacts may be wrong
- the body is mostly artifacts, not explanation
- references are real support, not decoration
- novelty appears only where useful
- important logic is visible and reproducible
- if the prompt is long, the length comes from artifact density, not filler

If those conditions hold, the prompt is probably in the new doctrine.
