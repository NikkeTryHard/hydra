# Hydra Prompt Style Guide

> [!WARNING]
> Do not hand-build large Hydra prompt packets.
> Use `scripts/generate_prompt.py`, then inspect the rendered prompt before you send it to an agent.
> The research agent is the most intelligent LLM on the planet, and the more artifacts you throw at it, the better it performs. Make sure you have squeezed as much LOC as possible for the prompt.

## 1. What this guide is for

This guide is for building good Hydra prompts with the current prompt generator.

It is not a giant doctrine dump.
It is not a prompt packet.
It is not a rulebook for copying old example prompts word-for-word.

The job of this file is simple:

- explain how to use `scripts/generate_prompt.py`
- explain how to choose and pack artifacts
- explain how to adapt template shells to the actual task
- explain how to keep prompts clear, dense, and useful

If a rule in here does not help you build a better prompt, cut the rule instead of preserving ceremony.

---

## 2. What the generator actually does

Main files:

- script: `scripts/generate_prompt.py`
- example config: `scripts/examples/prompt_config.example.json`
- tests: `scripts/tests/test_generate_prompt.py`

The generator is a small JSON-driven prompt assembly tool.
It is not a magical prompt framework.

What it does:

- loads a reference prompt shell from `shell_source_path`
- preserves that template's shell order and `<artifacts>` container placement
- preserves template section text by default when `shell_source_path` is used
- lets template-backed variants edit inherited shell text only through explicit `mode: "append"` or `mode: "delete"` entries
- lets non-template variants replace shell sections by tag
- lets variants append new shell sections when needed
- combines shared and per-variant artifacts
- renders artifact blocks with labels, explanations, source labels, fence language, and optional line numbers

Supported artifact kinds:

- `file_range`
- `file_full`
- `literal`

What it does not do:

- pick the right task framing for you
- decide what artifacts belong in the prompt
- guarantee the prompt is narrow enough
- guarantee the prompt is long enough
- guarantee the artifacts are correct
- replace manual review of the final rendered prompt

The generator helps you assemble prompts faster.
It does not replace prompt judgment.

---

## 3. Quickstart workflow

Typical flow:

1. start from `scripts/examples/prompt_config.example.json`
2. choose the closest reference example family
3. set `shell_source_path`
4. add reusable artifacts to the top-level `artifacts` list
5. add one or more `variants`
6. if a variant uses `shell_source_path`, keep inherited shell text by default and use explicit `mode: "append"` / `mode: "delete"` entries for edits
7. validate the config
8. generate the prompt
9. inspect the rendered result before using it

Useful commands:

```bash
python3 scripts/generate_prompt.py --config scripts/examples/prompt_config.example.json --list-variants
python3 scripts/generate_prompt.py --config scripts/examples/prompt_config.example.json --validate-only
python3 scripts/generate_prompt.py --config scripts/examples/prompt_config.example.json --variant narrow-focused
python3 scripts/generate_prompt.py --config scripts/examples/prompt_config.example.json --all-variants --output-dir /tmp/hydra-generated-prompts
```

If the rendered prompt is wrong, fix the config or the template choice.
Do not just shrug and ship it.

---

## 4. Reference examples are templates, not prisons

Reference examples are there to give you a good starting shell.
They are not rigid copy targets.

Use these example families as templates:

- `reference_prompt_example_001_narrow_focused.md`
- `reference_prompt_example_002_broad_novel_fuse_loop.md`
- `reference_prompt_example_003_balanced_narrow_not_overconstrained.md`

Use them like this:

- narrow implementation or validation task -> start from `reference_prompt_example_001_narrow_focused.md`
- broad novelty or cross-field exploration task -> start from `reference_prompt_example_002_broad_novel_fuse_loop.md`
- narrow task that still needs reasoning freedom -> start from `reference_prompt_example_003_balanced_narrow_not_overconstrained.md`

Important rule:

- keep what helps
- change what does not fit
- remove what conflicts
- add what the task actually needs

When a variant uses `shell_source_path`, the default should be to keep the example's `role`, `direction`, `style`, and other shell text.
Do not silently replace a whole inherited section just because you want to tweak it.

That means you should feel free to:

- append extra guidance to inherited `direction` or `style` when the task needs it
- delete inherited lines only when they are genuinely conflicting or harmful for the task
- add new instructions when the task needs more guidance
- remove conflicting or noisy wording when you can point to a real conflict

The generator already supports this.
For template-backed variants, shell edits must be explicit:

- use `mode: "append"` to add lines onto an inherited section
- use `mode: "delete"` to remove specific inherited lines or an entire inherited section

If you do not explicitly append or delete, the example text stays.

So treat the examples as maintained templates for structure, not as sacred prompt text.
They are strong defaults, and they should survive unless you have an explicit reason to change them.

---

## 5. Good prompting advice for Hydra work

### 5.1 Be explicit about the job

Say what the agent is trying to produce.

Usually that means one of:

- an implementation-ready blueprint
- a validation-ready blueprint
- a debugging or risk-audit blueprint

Do not ask for a memo when you want a buildable answer.

### 5.2 Keep the shell lean

The shell should usually be a few tight blocks such as:

- `role`
- `direction`
- `style`
- `artifact_note`

The shell should orient the task.
It should not become the task.

### 5.3 Put the real weight in the artifacts

Good Hydra prompts usually win or lose on artifact quality, not on shell cleverness.

Use the prompt body to carry:

- code
- docs
- tests
- formulas
- thresholds
- comments
- examples

The prompt should give the agent a strong starting packet, not force first-mile rediscovery from nothing.

### 5.4 Treat artifacts as evidence, not truth

The prompt should explicitly tell the agent that artifacts may be:

- stale
- partial
- inconsistent
- semantically wrong
- misleading by omission

The agent should inspect and critique them, not inherit them blindly.

### 5.5 Reduce ambiguity and conflicts

If instructions fight each other, fix the conflict before you generate the final prompt.

Bad prompt behavior often comes from:

- too many style bullets
- old inherited wording that no longer matches the task
- broad novelty instructions on narrow tasks
- output contracts that do not match the actual decision you want

Clear prompts beat crowded prompts.

### 5.6 Use examples as steering tools, not scripts

The example prompts are useful because they show good families of structure.
They are not a reason to copy every sentence.

Start from the closest family, then adapt.

### 5.7 Ask for visible reasoning when the task needs it

For important technical tasks, the prompt should push the agent to separate:

- direct artifact support
- external support
- inference
- proposal
- blocked or missing surface

Do this when it helps correctness.
Do not force giant reporting rituals for tiny tasks.

---

## 6. Conflict cleanup rules

Before shipping a prompt, clean up template conflicts.

If inherited template text does not match the task, do one of these:

- append clarifying task-specific guidance
- delete the conflicting inherited line(s)
- delete the whole inherited section only when it is genuinely the wrong shape for the task

Examples:

- a narrow local fix prompt should not inherit broad novelty language
- a practical repo task does not need giant “explore forever” filler if the task is already concrete
- a short scoped task does not need a bloated output ritual
- a hard research task may need extra instructions that the base template does not include

The goal is not to preserve every stock sentence.
The goal is to produce the strongest prompt for the actual task.

But the default bias is preserve-first, not rewrite-first.
The examples carry important style and direction pressure, so do not strip them down unless you can explain why.

Recommended bias:

- keep useful structure from the examples
- drop boilerplate that adds noise
- add missing constraints when they materially improve correctness

---

## 7. Artifact selection and packing

### 7.1 Prefer dense, useful evidence

Prefer:

- relevant code excerpts
- relevant docs
- tests that show current behavior
- formulas and thresholds when they matter
- short literal reminders only when they are genuinely useful

Avoid:

- decorative links with no context
- giant irrelevant file dumps
- filler artifacts that exist only to make the prompt look serious

### 7.2 Use the right artifact kind

- use `file_range` when the useful surface is local
- use `file_full` when the whole file matters
- use `literal` for short task-specific guidance or compact context blocks

### 7.3 Label and explain artifacts

An artifact block is much more useful when the agent can tell:

- what it is
- where it came from
- why it matters

Use `label`, `explanation`, and `source_label` for that.

### 7.4 Do not oversqueeze low-signal context

Pack in as much relevant context as needed.
Stop when extra context stops helping.

More context is good when it makes the task easier to ground.
More context is bad when it becomes:

- repetition
- stale doctrine spam
- irrelevant code
- conflicting instructions
- snippet overload that makes the agent guess harder instead of less

---

## 8. Long prompts are fine when justified

Large rendered prompts are normal for serious Hydra work.

The output prompt can be very large when the task needs it, including many thousands of lines and sometimes up to around 10k lines.

That is fine if the extra length is carrying real signal.

Good reasons for a long prompt:

- multiple code paths matter
- you need code plus tests plus docs together
- the method depends on formulas or paper excerpts
- the agent needs enough surrounding context to critique artifacts instead of pattern-matching isolated snippets

Bad reasons for a long prompt:

- repeated doctrine blocks
- decorative prose
- boilerplate copied from examples just because it was there
- redundant artifacts that do not sharpen the task

Good stopping rule:

- keep squeezing in relevant context while it clearly improves grounding
- stop when adding more context no longer helps enough to justify the noise or confusion risk

Do not chase prompt size for its own sake.

---

## 9. Recommended shell pattern

Most serious prompts can start from a shell like this:

```xml
<role>
Produce an implementation-ready blueprint.
Do not give a memo.
Your answer itself must be the blueprint.
</role>

<direction>
Work toward the strongest exact blueprint for [TASK].

We want a detailed answer that makes clear:
- [what the artifacts directly support]
- [what is only inference]
- [what should be kept, narrowed, deferred, or rejected]
- [how to implement or validate the surviving path with minimal guesswork]

Use the artifacts below to derive your conclusions.
</direction>

<style>
- no high-level survey
- no vague answer
- distinguish direct support from inference
- include formulas when needed
- include code-like detail when helpful
- keep the answer actionable and auditable
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit.
</artifact_note>

<artifacts>
...
</artifacts>
```

Treat this the same way as the reference examples: as a starting point.
Adapt it to the task.

---

## 10. Example config anatomy

The example config shows the intended pattern:

- `defaults` for shared title, shell defaults, and shared artifacts
- top-level `artifacts` as a reusable registry
- `variants` for task-specific prompt versions
- `shell_source_path` for the reference template family
- `shell_sections` for explicit template edits or non-template overrides
- `artifact_ids` for reusable artifact selection
- `output_file` for the rendered prompt path

Template-backed edit rule:

- if `shell_source_path` is set, inherited shell text stays unless a `shell_sections` entry explicitly declares `mode: "append"` or `mode: "delete"`
- do not assume same-tag replacement for template-backed variants
- use append most of the time; use delete only for real conflicts

The current example variants show three normal usage patterns:

- narrow-focused
- broad-fuse-loop
- balanced-narrow

Read `scripts/examples/prompt_config.example.json` when you want the quickest practical reminder of how the generator is meant to be used.

---

## 11. Final checklist

Before shipping a prompt, check:

- the chosen template family actually matches the task
- conflicting inherited instructions were removed or rewritten
- the shell is lean and clear
- the body carries the real evidence
- artifacts are framed as evidence, not truth
- prompt length comes from useful context, not filler
- the final rendered prompt was inspected before use

If those are true, the prompt is probably in good shape.
