# Hydra Prompt Style Guide

> [!WARNING]
> Do not hand-build big Hydra prompt packets.
> Use `scripts/generate_prompt.py`, then inspect rendered prompt before sending to agent.
> Do not stop after one packing pass. Render, inspect line count, add more high-signal artifacts, rerender, repeat until useful local context exhausted or task truly small.
> Research agent strong. More useful artifacts usually -> better output. For serious Hydra research, pack as much useful LOC as possible.

## 1. Good prompting advice for Hydra work

### 1.0 Decide whether the job is actually research-worthy before you build a giant packet

Do not spend Hydra research budget on question repo already answers after disciplined local triage.

Before any serious research-agent prompt, narrow lane locally first:

1. check authority docs in order
2. check relevant code and tests
3. check canonical/archive surfaces if authority docs stop short
4. decide whether real gap remains, and whether gap is unresolved semantics, missing evidence, or only engineering packaging

Rule of thumb:

- **research-worthy** = repo still does not settle core semantics, provenance, target object, trust object, or next-build blueprint after serious local search
- **not research-worthy** = lane already narrowed, target object clear enough, remaining work is build, validation, gating, or sequencing inside current repo reality

Important distinction:

- early triage may ask **"is this real next lane or not?"** once, only to avoid spending impl effort on fake path
- after lane already narrowed, stop spending research budget on **"does this exist?"**, **"is this real?"**, or **"can we build now?"** unless new evidence materially reopens semantics
- once lane survives triage, research prompt should usually ask for blueprint**:
  - what exact object to add
  - what files to change and in what order
  - what measurable gates or thresholds should control promotion
  - what would falsify lane and what fallback should be

Bad expensive-research prompts:

- "does this lane exist?"
- "can we build this now?"
- "is this maybe good idea?"

when local triage already narrowed lane enough to move to blueprint, validation, or impl planning.

Good research prompts after narrowing, when real semantic gap still remains:

- "what exact proof object should we add next?"
- "what is impl-ready blueprint?"
- "what acceptance criteria should control promotion or rejection?"
- "what is strongest fallback if this lane fails?"

Use research agent for hardest unresolved question, not to rerun existence checks after lane already narrowed.

### 1.1 Be explicit about the job

Say what agent must produce.

Usually one of:

- impl-ready blueprint
- validation-ready blueprint
- debugging or risk-audit blueprint

Do not ask for memo when you want buildable answer.

### 1.2 Keep the shell lean

Shell should usually be few tight blocks such as:

- `role`
- `task`
- `rules`
- `style`
- `artifact_note` when explicit evidence warning needed, or fold warning into `rules` when shell should stay tighter

Shell should orient task.
Shell should not become task.

Recommended split:

- `role` = who agent is for this prompt, short and customizable
- `task` = actual job and required deliverable, customizable per prompt
- `rules` = hard requirements, must-do / must-not-do behavior, tool/search/validation pressure
- `style` = softer presentation and reasoning guidance

### 1.3 Put the real weight in the artifacts

Good Hydra prompts usually win or lose on artifact quality, not shell cleverness.

Use prompt body to carry:

- code
- docs
- tests
- formulas
- thresholds
- comments
- examples

Prompt should give agent strong starting packet, not force first-mile rediscovery from nothing.

### 1.4 Treat artifacts as evidence, not truth

Prompt should explicitly tell agent artifacts may be:

- stale
- partial
- inconsistent
- semantically wrong
- misleading by omission

Agent should inspect and critique them, not inherit blindly.

### 1.5 Reduce ambiguity and conflicts

If instructions fight each other, fix conflict before generating final prompt.

Bad prompt behavior often comes from:

- too many style bullets
- old inherited wording no longer matching task
- broad novelty instructions on narrow tasks

## 2. What this guide is for

This guide is for building good Hydra prompts with current prompt generator.

It is not giant doctrine dump.
It is not prompt packet.
It is not rulebook for copying old example prompts word-for-word.

Job of this file is simple:

- explain how to use `scripts/generate_prompt.py`
- explain how to choose and pack artifacts
- explain how to adapt template shells to actual task
- explain how to keep prompts clear, dense, useful

If rule here does not help build better prompt, cut rule instead of preserving ceremony.
But do not casually strip instructions that increase search depth, validation pressure, or useful tool freedom because task is narrow.

---

## 3. What the generator actually does

Main files:

- script: `scripts/generate_prompt.py`
- example config: `scripts/examples/prompt_config.example.json`
- tests: `scripts/tests/test_generate_prompt.py`

Generator is small JSON-driven prompt assembly tool.
Not magical prompt framework.

What it does:

- loads reference prompt shell from `shell_source_path`
- preserves template's shell order and `<artifacts>` container placement
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

- pick right task framing for you
- decide what artifacts belong in prompt
- guarantee prompt narrow enough
- guarantee prompt long enough
- guarantee artifacts correct
- replace manual review of final rendered prompt

Generator helps assemble prompts faster.
It does not replace prompt judgment.

---

## 4. Quickstart workflow

Typical flow:

1. start from `scripts/examples/prompt_config.example.json`
2. choose closest reference example family
3. set `shell_source_path`
4. add reusable artifacts to top-level `artifacts` list
5. add one or more `variants`
6. if variant uses `shell_source_path`, keep inherited shell text by default and use explicit `mode: "append"` / `mode: "delete"` entries for edits
7. validate config
8. generate prompt and inspect reported line count
9. if prompt still looks light, do another packing pass with more high-signal local artifacts
10. rerender and repeat until prompt dense enough for task
11. inspect final rendered result before using it

Useful commands:

```bash
python3 scripts/generate_prompt.py --config scripts/examples/prompt_config.example.json --list-variants
python3 scripts/generate_prompt.py --config scripts/examples/prompt_config.example.json --validate-only
python3 scripts/generate_prompt.py --config scripts/examples/prompt_config.example.json --variant narrow-focused
python3 scripts/generate_prompt.py --config scripts/examples/prompt_config.example.json --all-variants --output-dir /tmp/hydra-generated-prompts
```

If rendered prompt is wrong, fix config or template choice.
Do not shrug and ship it.

If rendered prompt is suspiciously short for serious research task, that is not cosmetic.
Treat as packing failure until you either add more high-signal context or can explain exactly why task is truly narrow.

---

## 5. Reference examples are templates, not prisons

Reference examples exist to give good starting shell.
They are not rigid copy targets.

Use these example families as templates:

- `reference_prompt_example_001_narrow_focused.md`
- `reference_prompt_example_002_broad_novel_fuse_loop.md`
- `reference_prompt_example_003_balanced_narrow_not_overconstrained.md`

Use them like this:

- narrow impl or validation task -> start from `reference_prompt_example_001_narrow_focused.md`
- broad novelty or cross-field exploration task -> start from `reference_prompt_example_002_broad_novel_fuse_loop.md`
- narrow task that still needs reasoning freedom -> start from `reference_prompt_example_003_balanced_narrow_not_overconstrained.md`

Important rule:

- keep what helps
- change what does not fit
- remove what conflicts
- add what task needs

When variant uses `shell_source_path`, default should be to keep example's `role`, `task`, `rules`, `style`, and other shell text.
Do not silently replace whole inherited section because you want tweak.

That means feel free to:

- append extra guidance to inherited `role`, `task`, `rules`, or `style` when task needs it
- delete inherited lines only when genuinely conflicting or harmful for task
- add new instructions when task needs more guidance
- remove conflicting or noisy wording when you can point to real conflict

Generator already supports this.
For template-backed variants, shell edits must be explicit:

- use `mode: "append"` to add lines onto inherited section
- use `mode: "delete"` to remove specific inherited lines or whole inherited section

If you do not explicitly append or delete, example text stays.

So treat examples as maintained templates for structure, not sacred prompt text.
They are strong defaults, and should survive unless you have explicit reason to change them.

---

Clear prompts beat crowded prompts.

### 5.1 Use examples as steering tools, not scripts

Example prompts are useful because they show good structure families.
They are not reason to copy every sentence.

Start from closest family, then adapt.

### 5.2 Ask for visible reasoning when the task needs it

For important technical tasks, prompt should push agent to separate:

- direct artifact support
- external support
- inference
- proposal
- blocked or missing surface

Do this when it helps correctness.
Do not force giant reporting rituals for tiny tasks.

### 5.3 Preserve strong search, tool, and validation pressure by default

Some inherited style lines look repetitive, but for Hydra research they are usually load-bearing, not filler.

By default, preserve and reuse instructions in this family unless genuinely conflicting or impossible for task:

- tell agent to search broadly for papers, official docs, repos, and adjacent outside evidence when stronger external grounding could improve Hydra rather than merely describe Hydra's current state
- tell agent to prefer primary sources such as full papers or official docs over abstracts or shallow summaries whenever that evidence is available
- tell agent to use Python through bash tool not only for math, but also for quick parsers, data inspection, sanity checks, small experiments, validation scripts, benchmark arithmetic, and other lightweight research support work
- tell agent to include enough concrete detail that reviewer can validate, reproduce, or falsify answer later
- tell agent not to stop after first plausible pass when more discovery, validation, or tightening is still available inside scope

Why this matters:

- Hydra's goal is strength, not merely internal consistency with current repo snapshot
- stronger outside evidence can reveal better methods, failure cases, and tighter blueprints than local artifacts alone
- Python-in-bash often helps agent validate or sharpen claim even when task is not mainly mathematical
- explicit validation language makes it harder for agent to hide weak support behind polished prose
- anti-premature-stop loop pressure helps prevent “one quick pass and done” behavior on tasks that still benefit from more falsification or sharpening

If you delete one of these inherited instructions, be able to explain concrete conflict.
"This feels optional" is not strong enough by itself.

---

## 6. Conflict cleanup rules

Before shipping prompt, clean up template conflicts.

If inherited template text does not match task, do one of these:

- append clarifying task-specific guidance
- delete conflicting inherited line(s)
- delete whole inherited section only when genuinely wrong shape for task

Examples:

- narrow local fix prompt should not inherit broad novelty language
- practical repo task may not need broad novelty or cross-field fusion language if task already concrete
- but persistent search/validation/loop pressure is usually still useful and should only be removed when it clearly creates conflict, duplication, or obvious waste
- short scoped task does not need bloated output ritual
- hard research task may need extra instructions base template does not include

Goal is not preserve every stock sentence.
Goal is produce strongest prompt for actual task.

But default bias is preserve-first, not rewrite-first.
Examples carry important role/task/rules/style pressure, so do not strip them down unless you can explain why.

Recommended bias:

- keep useful structure from examples
- especially keep strong search, Python-tool freedom, validation-detail, and anti-premature-stop lines unless you can name concrete harm
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
- short literal reminders only when genuinely useful

Avoid:

- decorative links with no context
- giant irrelevant file dumps
- filler artifacts that exist only to make prompt look serious

### 7.2 Use the right artifact kind

- use `file_range` when useful surface is local
- use `file_full` when whole file matters
- use `literal` for short task-specific guidance or compact context blocks

### 7.3 Label and explain artifacts

Artifact block much more useful when agent can tell:

- what it is
- where it came from
- why it matters

Use `label`, `explanation`, and `source_label` for that.

### 7.4 Do not oversqueeze low-signal context

Pack in as much relevant context as needed.
Stop when extra context stops helping.

More context good when it makes task easier to ground.
More context bad when it becomes:

- repetition
- stale doctrine spam
- irrelevant code
- conflicting instructions
- snippet overload that makes agent guess harder instead of less

### 7.5 Under-packed serious research prompts are a failure mode

For serious Hydra research, usually better to overpack useful local evidence than underpack and force agent to rediscover obvious repo context.

Default bias:

- if multiple code paths, tests, doctrine layers, and archive surfaces are relevant, pack them
- if prompt for serious multi-surface research task renders suspiciously short, treat as warning sign and ask what high-signal local evidence is still missing
- squeeze in as much useful LOC as you can while added context still helps agent reason more accurately
- do not confuse prompt rendered successfully" with prompt is dense enough for Hydra-grade research"
- do not settle for first reasonable draft; do multiple packing passes by default on serious research tasks
- use generator's rendered line count and warning output as pressure to keep looking for more high-signal local evidence

Practical rule of thumb:

- few hundred lines may be fine for tiny local task
- but for major Hydra research lanes, prompt only few hundred lines is often under-packed unless task truly narrow and evidence surface genuinely small
- generated prompt under roughly 3000 lines should trigger another packing pass by default unless you can explain why task is genuinely small and already well-grounded

Target is not prompt size for its own sake.
Target is giving agent enough evidence that it does not have to spend first pass rediscovering repo reality you could have packed directly.

---

## 8. Long prompts are fine when justified

Large rendered prompts are normal for serious Hydra work.

Output prompt can be large when task needs it, including many thousands of lines and sometimes up to around 10k lines.

That is fine if extra length carries real signal.

Good reasons for long prompt:

- multiple code paths matter
- you need code plus tests plus docs together
- method depends on formulas or paper excerpts
- agent needs enough surrounding context to critique artifacts instead of pattern-matching isolated snippets
- you are trying to close semantically delicate Hydra lane and there are multiple live docs, source files, tests, and archive artifacts that all sharpen answer

Bad reasons for long prompt:

- repeated doctrine blocks
- decorative prose
- boilerplate copied from examples because it was there
- redundant artifacts that do not sharpen task

Good stopping rule:

- keep squeezing in relevant context while it clearly improves grounding
- stop when adding more context no longer helps enough to justify noise or confusion risk
- if unsure whether serious Hydra research prompt is dense enough, first move should usually be to look for more high-signal local artifacts before deciding it is done
- do at least one explicit repack-and-rerender pass after first generation for serious Hydra research prompts; do more when generator warning or your own review says packet still feels light

Do not chase prompt size for its own sake.

---

## 9. Example config anatomy

Example config shows intended pattern:

- `defaults` for shared title, shell defaults, and shared artifacts
- top-level `artifacts` as reusable registry
- `variants` for task-specific prompt versions
- `shell_source_path` for reference template family
- `shell_sections` for explicit template edits or non-template overrides
- `artifact_ids` for reusable artifact selection
- `output_file` for rendered prompt path

Template-backed edit rule:

- if `shell_source_path` is set, inherited shell text stays unless `shell_sections` entry explicitly declares `mode: "append"` or `mode: "delete"`
- do not assume same-tag replacement for template-backed variants
- use append most of time; use delete only for real conflicts

Current example variants show three normal usage patterns:

- narrow-focused
- broad-fuse-loop
- balanced-narrow

Read `scripts/examples/prompt_config.example.json` when you want quickest practical reminder of how generator is meant to be used.

Legacy note:

- old prompts may still use `direction`
- generator already supports arbitrary shell tags, so legacy prompts remain valid
- new examples should prefer `task` and use `rules` when you need real split between hard requirements and softer style pressure

---

## 10. Final checklist

Before shipping prompt, check:

- chosen template family matches task
- conflicting inherited instructions were removed or rewritten
- shell is lean and clear
- body carries real evidence
- artifacts are framed as evidence, not truth
- prompt length comes from useful context, not filler
- final rendered prompt was inspected before use

If those are true, prompt probably in good shape.