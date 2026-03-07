# Hydra prompt/answer run index

This directory now uses `research/agent_handoffs/combined_all_variants/` as the **single source of truth** for prompt/answer context artifacts.

## Canonical rule

Each preserved artifact or variant should have one self-contained markdown file under:

- `research/agent_handoffs/combined_all_variants/<artifact>.md`

Each file should put the prompt/context at the top and the answer/context note at the bottom, with explicit provenance tags.

## Why this exists

Hydra previously had prompt/answer artifacts spread across:

- `research/agent_handoffs/prompts/`
- `research/agent_handoffs/prior_answers/`
- `/agent_answers/`
- root-level `PROMPT_*.md`
- root-level `prompt-*.md`
- root-level `agent_*.md`

That made it too easy to lose track of which prompt produced which answer.

## Canonical combined artifacts

Examples now preserved under `combined_all_variants/`:

- direct follow-up prompt/answer pairs for runs 5 and 6
- the primary and revised run-7 variants
- the preserved substantive run-8 variants plus the later diagnostic note
- extracted mixed transcript records from `agent_answers/ANSWER_*.md`
- prompt-pack and prompt-template context records
- archived run-004 context preserved as a single combined file

## Non-SSOT materials

These folders are no longer intended to be the canonical prompt/answer source of truth:

- `runs/`
- `combined_runs/`
- `legacy/`
- `prior_answers/`
- `prompts/`
- `archival_runs_without_preserved_answers/`
- `prompt_packs/`

## Consolidation policy

Older folders may still exist temporarily for compatibility or cleanup staging, but **new work should not use them as the canonical archive**.

If the same prompt/answer information exists in both an older folder and `combined_all_variants/`, the `combined_all_variants/` copy wins.

## Rule for future work

When creating a new prompt/answer artifact:

1. create one self-contained markdown file under `combined_all_variants/`
2. put prompt/context at the top
3. put answer/context note at the bottom
4. tag provenance explicitly (`preserved`, `mixed_transcript`, `diagnostic`, `context_note`, etc.)
5. if an artifact is prompt-only or answer-only, say so explicitly rather than pretending it is a full run pair
